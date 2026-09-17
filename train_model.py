"""
train_model.py - Stream Processing AI Pokémon Trainer
Trains in batches, clears memory after each batch, saves to DB incrementally
"""

import os
import sys
import json
import logging
import sqlite3
import random
import time
import zipfile
import shutil
import subprocess
import gc
import resource
import threading
import queue
import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator
from io import BytesIO

# ============ NUCLEAR LOG SUPPRESSION ============
logging.root.handlers = []
logging.basicConfig = lambda *args, **kwargs: None

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except:
    pass

for name in logging.root.manager.loggerDict.keys():
    logging.getLogger(name).disabled = True
    logging.getLogger(name).setLevel(logging.CRITICAL)

sys.stderr = open(os.devnull, 'w') if not os.getenv("DEBUG") else sys.stderr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from tqdm import tqdm

# ============ SILENT LOGGER ============
class SilentLogger:
    def info(self, msg, *args, **kwargs):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} INFO {msg}")
    def warning(self, msg, *args, **kwargs):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} WARNING {msg}")
    def error(self, msg, *args, **kwargs):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} ERROR {msg}")
    def debug(self, msg, *args, **kwargs):
        pass

log = SilentLogger()

def log_memory(tag: str = ""):
    """
    Logs both:
    - peak RSS (ru_maxrss) - a HIGH-WATER MARK. By OS definition this can
      only stay flat or increase for the life of the process, no matter how
      well memory gets cleaned up afterward - a batch that transiently
      spiked memory once will keep this number elevated forever after,
      even if current usage drops right back down. This is why the old
      logs looked like a runaway leak even when cleanup was working.
    - current RSS (from /proc/self/status VmRSS, Linux-only, no extra
      dependency) - what's ACTUALLY resident right now. This is the number
      that actually predicts whether the next batch will get OOM-killed,
      and it's what _trim_memory() below is trying to keep low.
    """
    try:
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        current_mb = None
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        current_mb = int(line.split()[1]) / 1024
                        break
        except Exception:
            pass
        if current_mb is not None:
            log.info(f"   🧠 Memory{f' ({tag})' if tag else ''}: "
                     f"{current_mb:.0f} MB current RSS, {peak_mb:.0f} MB peak RSS")
        else:
            log.info(f"   🧠 Memory{f' ({tag})' if tag else ''}: {peak_mb:.0f} MB peak RSS")
    except Exception:
        pass


def _trim_memory():
    """
    Python/glibc's allocator frequently keeps freed heap pages around as
    reusable arenas instead of actually returning them to the OS, even
    after gc.collect() runs. That gap between "Python thinks it's freed
    this" and "the OS sees the memory back" is exactly what can push a
    <1GB container over its limit despite per-batch cleanup already being
    in place. malloc_trim(0) asks glibc to hand freed-but-retained pages
    back to the OS immediately. No-ops safely on non-glibc platforms.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class PrefetchIterator:
    """
    Wraps a slow iterator (e.g. one that hits the network) in a background
    thread with a small bounded queue, so the NEXT item can be fetched
    while the caller is busy doing something else (e.g. training on the
    current chunk) - overlapping network wait time with CPU compute time
    instead of paying both costs back-to-back.

    Deliberately conservative about not crashing the main process:
    - queue is bounded (maxsize) so a fast producer can't outrun the
      consumer and blow up memory
    - the background thread is a daemon, so it can never block shutdown
    - any exception in the producer is caught, stored, and re-raised in
      the MAIN thread on next() - it never crashes silently or hangs
    - if thread startup itself fails for any reason, falls back to
      plain synchronous iteration (no prefetching, but still works)
    """
    _SENTINEL = object()

    def __init__(self, source_iterable, maxsize: int = 4):
        self._source = source_iterable
        self._queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        try:
            self._thread = threading.Thread(target=self._produce, daemon=True)
            self._thread.start()
        except Exception as e:
            log.warning(f"   ⚠️ Prefetch thread failed to start ({e}), "
                        f"falling back to non-prefetched loading")
            self._thread = None

    def _produce(self):
        try:
            for item in self._source:
                self._queue.put(item)
        except BaseException as e:
            self._error = e
        finally:
            self._queue.put(self._SENTINEL)

    def __iter__(self):
        if self._thread is None:
            # Fallback: no thread running, just iterate the source directly.
            yield from self._source
            return
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                if self._error is not None:
                    raise self._error
                return
            yield item

# ============ CONFIGURATION ============

TURSO_URL = os.getenv("TURSO_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # Increased for streaming
STREAM_BATCH_SIZE = int(os.getenv("STREAM_BATCH_SIZE", "20"))  # Images per stream batch - kept small so only one small chunk is ever in memory at a time
EPOCHS = int(os.getenv("EPOCHS", "20"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
EARLY_STOP_ACC = float(os.getenv("EARLY_STOP_ACC", "95.0"))  # stop once train acc hits this %
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", "3"))  # stop if val acc doesn't improve for N epochs
DATASET_NAME = os.getenv("DATASET_NAME", "SpreadSheets/Poketwo-Spawn-Images")
MODEL_OUTPUT = os.getenv("MODEL_OUTPUT", "models/pokemon_classifier.pt")
DB_PATH = os.getenv("DB_PATH", "pokemon.db")
AUTO_EXTRACT_ARCHIVES = os.getenv("AUTO_EXTRACT_ARCHIVES", "true").lower() == "true"
MAX_SPECIES = int(os.getenv("MAX_SPECIES", "100"))  # Max species to train
MAX_IMAGES_PER_SPECIES = int(os.getenv("MAX_IMAGES_PER_SPECIES", "10"))
# Of each species' MAX_IMAGES_PER_SPECIES quota, this many are held out into
# a real validation set that NEVER gets trained on - see the note on the old
# validation code in stream_train() for why this matters.
VAL_IMAGES_PER_SPECIES = int(os.getenv("VAL_IMAGES_PER_SPECIES", "2"))
# Hard cap on how many images get cached in RAM for reuse across epochs.
# Each cached image is a 224x224x3 uint8 tensor (~150KB). Default 8000
# images caps the cache at ~1.2GB, which should be safe on most small
# Railway instances. Images beyond this cap still get used for training
# in epoch 1, they just aren't kept in memory - so if you hit the cap,
# epoch 2+ will train on a smaller (but still random/representative)
# subset instead of the full dataset. Raise this if your container has
# more RAM to spare; set to 0 to disable caching entirely (every epoch
# re-streams from Hugging Face - slower, but flat memory usage).
MAX_CACHE_IMAGES = int(os.getenv("MAX_CACHE_IMAGES", "0"))

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    log.info(f"🔑 HF_TOKEN configured")

DEVICE = torch.device("cpu")
torch.set_num_threads(os.cpu_count() or 4)

# ============ ARCHIVE EXTRACTION ============

def extract_archive_files():
    if not AUTO_EXTRACT_ARCHIVES:
        return
    
    archives = []
    for pattern in ["*.zip", "*.ZIP", "*.rar", "*.RAR", "*.7z", "*.7Z"]:
        archives.extend(Path(".").glob(pattern))
    
    if not archives:
        log.info("📦 No archive files found.")
        return
    
    log.info(f"📦 Found {len(archives)} archive file(s), extracting...")
    
    extra_dir = Path("Extra pokemons")
    extra_dir.mkdir(exist_ok=True)
    
    extracted_count = 0
    
    for archive_path in archives:
        try:
            ext = archive_path.suffix.lower()
            temp_dir = Path(f"temp_extract_{archive_path.stem}")
            temp_dir.mkdir(exist_ok=True)
            
            if ext in ['.zip']:
                log.info(f"   📂 Extracting ZIP: {archive_path.name}")
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            elif ext in ['.rar']:
                log.info(f"   📂 Extracting RAR: {archive_path.name}")
                try:
                    import rarfile
                    with rarfile.RarFile(archive_path) as rf:
                        rf.extractall(temp_dir)
                except:
                    subprocess.run(['unrar', 'x', '-y', str(archive_path), str(temp_dir)], 
                                 capture_output=True, check=False)
            elif ext in ['.7z']:
                log.info(f"   📂 Extracting 7z: {archive_path.name}")
                try:
                    import py7zr
                    with py7zr.SevenZipFile(archive_path, 'r') as sz:
                        sz.extractall(temp_dir)
                except:
                    subprocess.run(['7z', 'x', '-y', str(archive_path), f'-o{temp_dir}'], 
                                 capture_output=True, check=False)
            
            extracted = process_extracted_files(temp_dir, extra_dir)
            extracted_count += extracted
            
            shutil.rmtree(temp_dir)
            archive_path.unlink()
            log.info(f"   ✅ Extracted: {archive_path.name} ({extracted} images)")
            
        except Exception as e:
            log.error(f"   ❌ Failed to extract {archive_path}: {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    if extra_dir.exists():
        species = [f for f in extra_dir.iterdir() if f.is_dir()]
        if species:
            log.info(f"   📁 Extracted {len(species)} species, {extracted_count} images total")


def process_extracted_files(temp_dir: Path, extra_dir: Path) -> int:
    extracted_count = 0
    valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.PNG', '.JPG', '.JPEG', '.WEBP'}
    
    for root, dirs, files in os.walk(temp_dir):
        root_path = Path(root)
        images = [f for f in files if Path(f).suffix in valid_extensions]
        
        if images:
            rel_path = root_path.relative_to(temp_dir)
            # BUG FIXED: was `rel_path.parts[0]`, which grabs the FIRST path
            # component under temp_dir. If the zip has a wrapper folder
            # around the real per-species folders (e.g. "Extra pokemons/DJ
            # Rotom/img.jpg"), parts[0] is "Extra pokemons" - the wrapper,
            # not the species - so every species collapses into one fake
            # "species" (this is exactly why the log showed "Extracted 1
            # species, 153 images total" instead of dozens). parts[-1] is
            # the immediate parent folder of the images - the actual
            # species folder - regardless of how many wrapper folders sit
            # above it.
            species_name = rel_path.parts[-1].replace("_", " ").strip() if rel_path.parts else "unknown"
            
            dest_dir = extra_dir / species_name.replace(" ", "_")
            dest_dir.mkdir(exist_ok=True)
            
            for img_name in images:
                src = root_path / img_name
                dest = dest_dir / img_name
                if dest.exists():
                    counter = 1
                    stem = Path(img_name).stem
                    suffix = Path(img_name).suffix
                    while dest.exists():
                        dest = dest_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                shutil.move(str(src), str(dest))
                extracted_count += 1
    
    return extracted_count

# ============ DATABASE LAYER ============

class Database:
    def __init__(self):
        self.use_turso = False
        self._conn = None
        
        if TURSO_URL:
            try:
                import libsql
                self._conn = libsql.connect(TURSO_URL, auth_token=TURSO_AUTH_TOKEN)
                self.use_turso = True
                log.info(f"✅ Connected to Turso database")
            except Exception:
                log.warning(f"Turso connection failed, using SQLite fallback")
        
        if not self.use_turso:
            self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            log.info(f"✅ Using SQLite database: {DB_PATH}")
        
        self._create_tables()
    
    def _create_tables(self):
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pokemon_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species TEXT NOT NULL,
                variant_name TEXT NOT NULL,
                feature_vector TEXT NOT NULL,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                UNIQUE(species, variant_name)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS species_info (
                species TEXT PRIMARY KEY,
                count INTEGER DEFAULT 0,
                last_updated INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)
        self._conn.commit()
        log.info("✅ Database tables ready")
    
    def add_pokemon_features(self, species: str, features: List[np.ndarray], 
                             variant_names: List[str] = None):
        self.add_pokemon_features_batch({species: features})

    def add_pokemon_features_batch(self, species_to_features: Dict[str, List[np.ndarray]]):
        """
        Writes features for MULTIPLE species in ONE cursor + ONE commit,
        instead of one cursor/commit per species. This matters a lot once
        we're saving after every small training chunk (a chunk of ~25
        images can easily span 15-25 different species) - doing a separate
        commit per species per chunk multiplies DB round-trips a lot, and
        with the experimental Turso client in particular that repeated
        cursor/commit churn is a plausible source of memory growth over
        many batches. Batching it into one commit per chunk avoids that.

        variant_name is offset by each species' EXISTING row count so
        chunks accumulate (species_11, species_12, ...) instead of every
        chunk restarting at species_1 and silently INSERT-OR-REPLACE-ing
        over an earlier chunk's rows for the same species.
        """
        if not species_to_features:
            return
        cursor = self._conn.cursor()
        try:
            for species, features in species_to_features.items():
                if not features:
                    continue
                cursor.execute(
                    "SELECT COUNT(*) FROM pokemon_features WHERE species = ?", (species,)
                )
                existing_count = cursor.fetchone()[0]
                variant_names = [f"{species}_{existing_count + i + 1}" for i in range(len(features))]
                for i, feature in enumerate(features):
                    feature_json = json.dumps(feature.tolist())
                    cursor.execute("""
                        INSERT OR REPLACE INTO pokemon_features 
                        (species, variant_name, feature_vector, created_at)
                        VALUES (?, ?, ?, strftime('%s', 'now'))
                    """, (species, variant_names[i], feature_json))
                cursor.execute("""
                    INSERT INTO species_info (species, count, last_updated)
                    VALUES (?, ?, strftime('%s', 'now'))
                    ON CONFLICT(species) DO UPDATE SET
                        count = count + excluded.count,
                        last_updated = excluded.last_updated
                """, (species, len(features)))
            self._conn.commit()
        finally:
            try:
                cursor.close()
            except Exception:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM pokemon_features")
        total_features = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM species_info")
        total_species = cursor.fetchone()[0]
        return {"total_features": total_features, "total_species": total_species, "use_turso": self.use_turso}

    def save_checkpoint_blob(self, data: bytes):
        """
        Stores the full training checkpoint (model + optimizer + scheduler
        state) in the training_metadata table, split into ~400KB chunks
        rather than one single row.

        BUG FIXED: the previous version wrote the whole (multi-MB, base64-
        encoded) checkpoint as ONE row. Turso's hosted service commonly
        rejects or truncates large single-row writes - and the training
        loop only ever caught that as a generic warning and moved on, so a
        save could silently fail every single time with no clear signal in
        the logs. That's the most likely explanation for "still crashed and
        started from the beginning" even after checkpointing was added.
        Splitting into small chunks avoids per-row size limits, and
        reading the write back immediately (below) turns a silent failure
        into a loud, specific one.
        """
        CHUNK_SIZE = 400_000  # raw bytes per chunk, before base64 (~533KB encoded)
        cursor = self._conn.cursor()
        try:
            chunks = [data[i:i + CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)] or [b""]
            for i, chunk in enumerate(chunks):
                b64 = base64.b64encode(chunk).decode("ascii")
                cursor.execute("""
                    INSERT OR REPLACE INTO training_metadata (key, value, updated_at)
                    VALUES (?, ?, strftime('%s', 'now'))
                """, (f"checkpoint_chunk_{i:04d}", b64))
            # Meta row last, so a load never sees a meta count higher than
            # the chunks actually written (e.g. if this call dies partway).
            cursor.execute("""
                INSERT OR REPLACE INTO training_metadata (key, value, updated_at)
                VALUES ('checkpoint_meta', ?, strftime('%s', 'now'))
            """, (json.dumps({"chunks": len(chunks), "total_bytes": len(data)}),))
            # Clean up any leftover chunks from a PREVIOUS checkpoint that
            # had more chunks than this one (checkpoint size can shrink -
            # e.g. optimizer state changes) - otherwise a load would
            # wrongly append stale trailing bytes from an old save.
            cursor.execute("""
                SELECT key FROM training_metadata 
                WHERE key LIKE 'checkpoint_chunk_%' AND key NOT IN ({})
            """.format(",".join("?" * len(chunks))), tuple(f"checkpoint_chunk_{i:04d}" for i in range(len(chunks))))
            stale_keys = [row[0] for row in cursor.fetchall()]
            for key in stale_keys:
                cursor.execute("DELETE FROM training_metadata WHERE key = ?", (key,))
            self._conn.commit()
        finally:
            try:
                cursor.close()
            except Exception:
                pass

        # Read-after-write verification - turns a silent DB-side rejection
        # into a loud, immediate, specific failure instead of a mysterious
        # "why did it start from scratch again" three restarts from now.
        verify = self.load_checkpoint_blob()
        if verify is None or len(verify) != len(data):
            got = 0 if verify is None else len(verify)
            raise RuntimeError(
                f"Checkpoint verification failed: wrote {len(data)} bytes, "
                f"read back {got} bytes. The DB write likely did not "
                f"actually persist (e.g. a Turso row/size limit)."
            )

    def load_checkpoint_blob(self) -> Optional[bytes]:
        cursor = self._conn.cursor()
        try:
            cursor.execute("SELECT value FROM training_metadata WHERE key = 'checkpoint_meta'")
            row = cursor.fetchone()
            if not row or not row[0]:
                return None
            meta = json.loads(row[0])
            num_chunks = meta.get("chunks", 0)
            parts = []
            for i in range(num_chunks):
                cursor.execute(
                    "SELECT value FROM training_metadata WHERE key = ?",
                    (f"checkpoint_chunk_{i:04d}",)
                )
                r = cursor.fetchone()
                if not r or not r[0]:
                    log.warning(f"   ⚠️ Checkpoint chunk {i}/{num_chunks} missing from DB")
                    return None
                parts.append(base64.b64decode(r[0]))
            return b"".join(parts)
        finally:
            try:
                cursor.close()
            except Exception:
                pass

    def close(self):
        if self._conn:
            self._conn.close()


# ============ AI MODEL ============

class PokemonFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        # ShuffleNetV2 x0.5: fastest of the reasonable CPU-friendly options
        # (1.4M params vs MobileNetV3-Small's 2.5M) - swapped in specifically
        # for epoch speed now that the disk cache means epochs 2+ are
        # compute-bound rather than network-bound. Trade-off: its features
        # are somewhat weaker than MobileNetV3-Small's, which matters more
        # than usual here given how little data there is per species (25
        # images x 1119 classes) - worth watching train/val accuracy after
        # this swap to confirm it's not costing more than the speed is worth.
        self.backbone = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        backbone_dim = 1024  # ShuffleNetV2 x0.5's pre-fc feature dim
        
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        self.to(DEVICE)
        self.eval()
    
    @torch.no_grad()
    def extract(self, img: Image.Image) -> np.ndarray:
        if img is None:
            return np.zeros(256)
        
        try:
            img_tensor = self.transform(img).unsqueeze(0).to(DEVICE)
            img_tensor = self.normalize(img_tensor)
            # backbone already returns a flat (N, backbone_dim) vector —
            # its forward() does avgpool+flatten internally before the
            # (now-identity) classifier. Do NOT re-pool here; the old
            # code's adaptive_avg_pool2d on an already-2D tensor was
            # silently throwing and falling back to an all-zero vector
            # on every single image.
            features = self.backbone(img_tensor)
            projected = self.projection(features)
            projected = F.normalize(projected, p=2, dim=1)
            return projected.cpu().numpy().flatten()
        except Exception as e:
            log.warning(f"   ⚠️ Feature extraction failed: {e}")
            return np.zeros(256)
    
    def extract_batch(self, images: List[Image.Image], grad: bool = False) -> np.ndarray:
        """
        grad=False (default): used everywhere at inference time (bot lookups,
        DB feature snapshots) - whole thing runs under no_grad, nothing trains.

        grad=True: used during training. The backbone is still frozen/no_grad
        (it's pretrained ImageNet weights we don't want to disturb), but
        `projection` runs WITH grad so its weights actually receive gradient
        updates. Previously this whole method - backbone AND projection - ran
        under @torch.no_grad(), so `projection` never trained and stayed at
        its random init for the entire run while only the final classifier
        head learned on top of that noise. That's a big part of why accuracy
        was stuck low.
        """
        valid_images = []
        for img in images:
            if img is not None and isinstance(img, Image.Image):
                try:
                    valid_images.append(self.transform(img).unsqueeze(0))
                except Exception:
                    continue
        
        if not valid_images:
            return np.zeros((len(images), 256))
        
        try:
            batch_tensor = torch.cat(valid_images, dim=0).to(DEVICE)
            batch_tensor = self.normalize(batch_tensor)

            with torch.no_grad():
                features = self.backbone(batch_tensor)  # frozen, already flat (N, backbone_dim)

            if grad:
                projected = self.projection(features)
            else:
                with torch.no_grad():
                    projected = self.projection(features)
            projected = F.normalize(projected, p=2, dim=1)

            if grad:
                # keep it a tensor with grad history for the training loop
                result = projected
                if len(valid_images) < len(images):
                    pad = torch.zeros(len(images) - len(valid_images), result.shape[1], device=result.device)
                    result = torch.cat([result, pad], dim=0)
                return result

            result = projected.cpu().numpy()
            if len(valid_images) < len(images):
                padded = np.zeros((len(images), result.shape[1]))
                padded[:len(valid_images)] = result
                return padded
            return result
        except Exception as e:
            log.warning(f"   ⚠️ Batch feature extraction failed: {e}")
            if grad:
                return torch.zeros(len(images), 256, device=DEVICE, requires_grad=True)
            return np.zeros((len(images), 256))


class PokemonClassifier(nn.Module):
    def __init__(self, num_species: int):
        super().__init__()
        self.feature_extractor = PokemonFeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_species)
        )
    
    def forward_batch(self, images: List[Image.Image], grad: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor.extract_batch(images, grad=grad)
        if grad:
            # already a tensor with grad history - don't detach it via torch.tensor()
            features_tensor = features.to(DEVICE)
        else:
            features_tensor = torch.tensor(features, dtype=torch.float32).to(DEVICE)
        logits = self.classifier(features_tensor)
        return features_tensor, logits


# ============ STREAMING DATASET ============

class StreamingPokemonDataset(IterableDataset):
    """
    Streams images from Hugging Face in batches.
    Processes STREAM_BATCH_SIZE images, trains, then clears memory.
    """
    
    def __init__(self, extra_dir: str = "Extra pokemons"):
        self.extra_dir = extra_dir
        # NOTE: intentionally stops short of ToTensor+Normalize here. Every
        # image produced by this transform gets cached in self._cache for
        # reuse across epochs (see __iter__), so we store it as a uint8
        # tensor (raw 0-255 pixels) instead of a normalized float32 tensor.
        # That's a 4x memory cut (1 byte/channel vs 4 bytes/channel) with
        # ~28k images this is the difference between ~16.8GB and ~4.2GB
        # of cache. Normalization is applied later, per-batch, right before
        # the forward pass (see _to_normalized_float below).
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.PILToTensor(),  # uint8, CHW, values 0-255
        ])
        
        # Cache of (tensor, label) collected on the first pass, reused for
        # every later epoch so we never re-hit Hugging Face after epoch 1.
        self._cache: List[Tuple[torch.Tensor, int]] = []
        self._cache_ready = False
        
        # Disk-backed chunk cache: epoch 1 writes (tensor,label) pairs to
        # small files on disk as it streams from HF, instead of only
        # keeping them in RAM. Epoch 2+ then reads those chunk files
        # straight off disk - no network calls, no RAM buildup (one small
        # chunk loaded at a time), and it even survives a container
        # restart. This is independent of MAX_CACHE_IMAGES (the RAM cap)
        # and is the main lever for avoiding a slow multi-hour re-scan of
        # Hugging Face on every epoch.
        self.disk_cache_enabled = os.getenv("DISK_CACHE", "true").lower() == "true"
        self.disk_cache_dir = Path(os.getenv("DISK_CACHE_DIR", "image_cache"))
        self._disk_chunk_files: List[Path] = []
        self._resume_partial_chunks: List[Path] = []
        self._complete_marker = self.disk_cache_dir / "_complete.marker"
        if self.disk_cache_enabled:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
            existing = sorted(self.disk_cache_dir.glob("chunk_*.pt"))
            if existing and self._complete_marker.exists():
                # A previous run finished its full streaming pass (hit quota
                # or exhausted HF) and said so explicitly - safe to treat
                # this as the whole dataset and never touch HF again.
                self._disk_chunk_files = existing
                self._cache_ready = True
                log.info(f"   💽 Found {len(existing)} cached chunk files on disk "
                         f"from a previous COMPLETED run - will replay from disk, no HF re-stream")
            elif existing:
                # BUG FIXED: chunk files with no completion marker means the
                # previous run was cut off mid-stream (OOM, restart, redeploy)
                # before finishing. The old code couldn't tell this apart from
                # a genuinely finished run and would lock onto this partial
                # slice as "the dataset" forever, silently starving training
                # of the rest of the data on every future run. Now: replay
                # what's already saved (so it isn't wasted) and keep
                # streaming from HF afterward to fill out the remaining quota.
                self._resume_partial_chunks = existing
                log.info(f"   💽 Found {len(existing)} PARTIAL cached chunk files "
                         f"(no completion marker - a previous run was cut off) - "
                         f"will replay those, then resume streaming from Hugging "
                         f"Face to fill the rest")
        
        # Held-out validation images - collected once during the first
        # streaming pass (first VAL_IMAGES_PER_SPECIES images of each
        # species), never yielded for training, persisted to its own file
        # so it survives restarts just like the train chunks do.
        self._val_cache: List[Tuple[torch.Tensor, int]] = []
        self.val_cache_path = self.disk_cache_dir / "val_cache.pt"
        if self.disk_cache_enabled and self.val_cache_path.exists():
            try:
                self._val_cache = torch.load(self.val_cache_path, weights_only=False)
                log.info(f"   💽 Loaded {len(self._val_cache)} held-out validation images from disk")
            except Exception as e:
                log.warning(f"   ⚠️ Failed to load validation cache: {e}")

        # Build species mapping FIRST
        self._build_species_mapping()
        
        log.info(f"📊 Species mapping built: {len(self.species_to_idx)} species")
    
    def _build_species_mapping(self):
        """Build species mapping from Hugging Face + local extras."""
        
        # Get species from local extras
        local_species = set()
        extra_path = Path(self.extra_dir)
        if extra_path.exists():
            for folder in extra_path.iterdir():
                if folder.is_dir():
                    species = folder.name.replace("_", " ").strip().lower()
                    if species:
                        local_species.add(species)
        
        # Try to get species list from Hugging Face
        hf_species = set()
        try:
            from datasets import load_dataset
            
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                
                try:
                    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
            
            # Detect label column
            features = ds.features
            label_col = None
            for col in ["label", "text", "name", "species", "pokemon"]:
                if col in features:
                    label_col = col
                    break
            
            if label_col:
                # Fast path: if this is a HF ClassLabel column, the full
                # species list is already in the schema — no need to scan
                # rows at all. With 1,150 species, scanning rows to find
                # them (the old approach, capped at 500 rows) would miss
                # most species regardless of MAX_SPECIES.
                col_feature = features[label_col]
                names = getattr(col_feature, "names", None)
                if names:
                    hf_species = {str(n).strip().lower() for n in names}
                    log.info(f"   📋 Got {len(hf_species)} species directly from dataset schema")
                else:
                    # Fallback: scan rows. Raised from 500 -> 20000 with a
                    # heartbeat so it doesn't look frozen, but the schema
                    # path above should be what actually fires.
                    count = 0
                    for row in ds:
                        raw_label = row[label_col]
                        if isinstance(raw_label, int):
                            raw_label = features[label_col].int2str(raw_label)
                        species = str(raw_label).strip().lower()
                        hf_species.add(species)
                        count += 1
                        if count % 2000 == 0:
                            log.info(f"   🔎 Scanned {count} rows, found {len(hf_species)} species so far")
                        if count > 20000:
                            break
        except Exception as e:
            log.warning(f"Could not get species from Hugging Face: {e}")
        
        # Combine species
        all_species = sorted(local_species | hf_species)
        
        if not all_species:
            log.error("❌ No species found!")
            return
        
        self.species_to_idx = {s: i for i, s in enumerate(all_species)}
        self.idx_to_species = {i: s for s, i in self.species_to_idx.items()}
        
        # Limit species if needed
        if MAX_SPECIES > 0 and len(all_species) > MAX_SPECIES:
            # Keep local species first, then add from HF
            local_list = list(local_species)
            hf_list = [s for s in all_species if s not in local_species]
            selected = local_list + hf_list[:MAX_SPECIES - len(local_list)]
            self.species_to_idx = {s: i for i, s in enumerate(selected)}
            self.idx_to_species = {i: s for s, i in self.species_to_idx.items()}
            log.info(f"   Limited to {len(selected)} species (MAX_SPECIES={MAX_SPECIES})")
    
    def __iter__(self) -> Iterator:
        """
        Epoch 1: streams local images + Hugging Face images, caching every
        yielded (tensor, label) pair. Stops hitting HF as soon as every
        species has MAX_IMAGES_PER_SPECIES images (instead of scanning the
        whole remaining dataset for nothing).
        Epoch 2+: replays from the in-memory cache, no network at all.
        """
        
        if self._disk_chunk_files:
            log.info(f"   💽 Replaying {len(self._disk_chunk_files)} chunk files from disk "
                     f"(no HF re-stream, no network)")
            for chunk_path in self._disk_chunk_files:
                try:
                    chunk = torch.load(chunk_path, weights_only=False)
                    for item in chunk:
                        yield item
                    del chunk
                except Exception as e:
                    log.warning(f"   ⚠️ Failed to load cache chunk {chunk_path}: {e}")
            gc.collect()
            _trim_memory()
            return
        
        if self._cache_ready:
            log.info(f"   ♻️  Replaying {len(self._cache)} cached images (no HF re-stream)")
            for item in self._cache:
                yield item
            return
        
        target_total = len(self.species_to_idx) * MAX_IMAGES_PER_SPECIES
        species_counts: Dict[str, int] = {}
        collected = 0
        cache_capped_warned = False
        t_start = time.time()
        
        # Buffer for the disk-backed cache - flushed to a new chunk file
        # every DISK_CHUNK_SIZE images so we're never holding more than
        # one chunk's worth in RAM for this purpose.
        DISK_CHUNK_SIZE = 200
        disk_buffer: List[Tuple[torch.Tensor, int]] = []
        chunk_idx = 0

        # Resume from a PARTIAL disk cache left by a run that got cut off:
        # replay it first (so that progress isn't wasted and we don't
        # re-download images we already have) and update species_counts/
        # collected so the HF stream below only fetches what's still
        # missing, instead of starting the whole quota over from zero.
        if self._resume_partial_chunks:
            chunk_idx = len(self._resume_partial_chunks)
            replayed = 0
            for chunk_path in self._resume_partial_chunks:
                try:
                    chunk = torch.load(chunk_path, weights_only=False)
                    for tensor, label in chunk:
                        species = self.idx_to_species.get(label)
                        if species:
                            species_counts[species] = species_counts.get(species, 0) + 1
                            collected += 1
                            replayed += 1
                        yield tensor, label
                    del chunk
                except Exception as e:
                    log.warning(f"   ⚠️ Failed to replay partial chunk {chunk_path}: {e}")
            gc.collect()
            _trim_memory()
            log.info(f"   💽 Replayed {replayed} images from partial cache "
                     f"({collected}/{target_total}) - resuming HF stream for the rest")

        def _flush_disk_chunk():
            nonlocal disk_buffer, chunk_idx
            if not self.disk_cache_enabled or not disk_buffer:
                return
            chunk_path = self.disk_cache_dir / f"chunk_{chunk_idx:05d}.pt"
            try:
                torch.save(disk_buffer, chunk_path)
                self._disk_chunk_files.append(chunk_path)
                chunk_idx += 1
                # Re-save the (small) held-out val set alongside each train
                # chunk flush too, so a crash mid-stream doesn't lose
                # whatever validation images were collected so far.
                if self._val_cache:
                    torch.save(self._val_cache, self.val_cache_path)
            except Exception as e:
                log.warning(f"   ⚠️ Failed to write cache chunk to disk: {e}")
            disk_buffer = []
            gc.collect()
            _trim_memory()
        
        def _emit(tensor, label, species):
            nonlocal collected, cache_capped_warned
            # Position of this image within its species' quota, BEFORE
            # incrementing - used to decide train vs. held-out validation.
            species_idx = species_counts.get(species, 0)
            species_counts[species] = species_idx + 1
            collected += 1

            if species_idx < VAL_IMAGES_PER_SPECIES:
                # Held out - never trained on, never disk-chunked into the
                # training cache. Returning None tells the caller not to
                # yield this one into the training stream.
                self._val_cache.append((tensor, label))
                return None

            # Cap cache growth so a large run can't OOM the container.
            # Images beyond the cap still get trained on this epoch (via
            # the yield below), they just aren't retained for epoch 2+.
            #
            # BUG FIXED: `MAX_CACHE_IMAGES <= 0 or ...` used to mean "0 = no
            # limit" (the opposite of what the config comment above promises
            # - "set to 0 to disable caching entirely"). With the default of
            # 0 this appended EVERY streamed image to an unbounded in-memory
            # list - on a ~27,975-image run that's several GB of uint8
            # tensors, on top of the disk-backed chunk cache already doing
            # the real epoch-2+ replay job below. That's what was causing
            # the OOM kill. Now: when disk caching is on (the default),
            # skip the in-RAM copy entirely since disk chunks already do
            # cross-epoch replay. When disk caching is off, 0 actually
            # disables the RAM cache as documented; a positive value caps it.
            if self.disk_cache_enabled:
                pass
            elif MAX_CACHE_IMAGES > 0 and len(self._cache) < MAX_CACHE_IMAGES:
                self._cache.append((tensor, label))
            elif MAX_CACHE_IMAGES > 0 and not cache_capped_warned:
                cache_capped_warned = True
                log.warning(f"   ⚠️ Cache cap ({MAX_CACHE_IMAGES} images) reached - "
                            f"later epochs will replay only the cached subset, "
                            f"not the full {target_total}-image target. Raise "
                            f"MAX_CACHE_IMAGES if you have RAM to spare.")
            if self.disk_cache_enabled:
                disk_buffer.append((tensor, label))
                if len(disk_buffer) >= DISK_CHUNK_SIZE:
                    _flush_disk_chunk()
            return tensor, label
        
        # First, yield local images
        extra_path = Path(self.extra_dir)
        if extra_path.exists():
            for folder in extra_path.iterdir():
                if not folder.is_dir():
                    continue
                
                species = folder.name.replace("_", " ").strip().lower()
                if species not in self.species_to_idx:
                    continue
                
                label = self.species_to_idx[species]
                valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}
                
                for img_path in folder.iterdir():
                    if species_counts.get(species, 0) >= MAX_IMAGES_PER_SPECIES:
                        break
                    if img_path.suffix.lower() not in valid_extensions:
                        continue
                    
                    try:
                        img = Image.open(img_path).convert("RGB")
                        if img.size[0] > 10 and img.size[1] > 10:
                            item = _emit(self.transform(img), label, species)
                            if item is not None:
                                yield item
                    except Exception:
                        continue
        
        log.info(f"   📂 Local images collected: {collected}/{target_total}")
        
        if collected >= target_total:
            _flush_disk_chunk()
            self._cache_ready = True
            self._write_completion_marker(collected, target_total)
            log.info(f"   ✅ Quota already met from local images, skipping HF stream")
            return
        
        # Then, stream from Hugging Face
        stream_error = False
        try:
            from datasets import load_dataset
            
            ds = load_dataset(DATASET_NAME, split="train", streaming=True)
            
            # Detect columns
            features = ds.features
            label_col = None
            image_col = None
            
            for col in ["label", "text", "name", "species", "pokemon"]:
                if col in features:
                    label_col = col
                    break
            for col in ["image", "img", "picture"]:
                if col in features:
                    image_col = col
                    break
            
            if not label_col or not image_col:
                label_col = list(features.keys())[0]
                image_col = list(features.keys())[1] if len(features) > 1 else list(features.keys())[0]
            
            rows_scanned = 0
            
            for row in ds:
                rows_scanned += 1
                
                # Heartbeat so it never looks frozen, even mid-scan
                if rows_scanned % 200 == 0:
                    elapsed = time.time() - t_start
                    log.info(f"   🔎 Scanned {rows_scanned} HF rows, "
                             f"kept {collected}/{target_total} images "
                             f"({elapsed:.0f}s elapsed)")
                
                try:
                    raw_label = row[label_col]
                    if isinstance(raw_label, int):
                        raw_label = features[label_col].int2str(raw_label)
                    
                    species = str(raw_label).strip().lower()
                    
                    if species not in self.species_to_idx:
                        continue
                    
                    if species_counts.get(species, 0) >= MAX_IMAGES_PER_SPECIES:
                        continue
                    
                    img = row[image_col]
                    if img is None:
                        continue
                    
                    if not isinstance(img, Image.Image):
                        img = Image.open(BytesIO(img))
                    
                    if img.size[0] < 10 or img.size[1] < 10:
                        continue
                    
                    label = self.species_to_idx[species]
                    
                    item = _emit(self.transform(img), label, species)
                    if item is not None:
                        yield item
                    
                    # Stop as soon as every species has its quota instead
                    # of scanning the rest of the dataset for nothing.
                    if collected >= target_total:
                        log.info(f"   ✅ Quota met ({collected}/{target_total}) "
                                 f"after scanning {rows_scanned} HF rows")
                        break
                    
                except Exception:
                    continue
                
        except Exception as e:
            log.warning(f"Error streaming from Hugging Face: {e}")
            stream_error = True
        
        if collected < target_total:
            log.warning(f"   ⚠️ Only found {collected}/{target_total} images "
                        f"before HF dataset was exhausted")
        
        _flush_disk_chunk()
        self._cache_ready = True
        # Only mark the disk cache "complete" (skip HF forever on future
        # runs) if we actually finished the pass - quota met, or the HF
        # iterator ran out cleanly. If it stopped because of a raised
        # exception (network drop etc.) leave it unmarked so a restart
        # resumes the partial-chunk path above and keeps trying to fill
        # the quota, instead of silently freezing on whatever was collected
        # right before the error.
        if not stream_error:
            self._write_completion_marker(collected, target_total)
    
    def _write_completion_marker(self, collected: int, target_total: int):
        if not self.disk_cache_enabled:
            return
        try:
            self._complete_marker.write_text(json.dumps({
                "collected": collected,
                "target_total": target_total,
                "species": len(self.species_to_idx),
                "completed_at": time.time(),
            }))
        except Exception as e:
            log.warning(f"   ⚠️ Failed to write cache-completion marker: {e}")

    def get_val_set(self) -> List[Tuple[torch.Tensor, int]]:
        return self._val_cache

    def get_num_species(self) -> int:
        return len(self.species_to_idx)


# ============ STREAMING TRAINER ============

def store_chunk_features_to_db(features_tensor, batch_labels, idx_to_species: Dict[int, str], db):
    """
    Saves the features for a chunk that was JUST trained on, straight to
    the DB, using the features already computed during that chunk's
    forward pass (no extra model call needed). Called right after every
    small chunk so results are persisted and the chunk can be dropped
    from memory immediately - nothing waits until end-of-epoch.
    """
    feature_buffer: Dict[str, List[np.ndarray]] = {}
    feats_np = features_tensor.detach().cpu().numpy()
    labels_list = batch_labels.detach().cpu().tolist()
    for feat, lbl in zip(feats_np, labels_list):
        if not np.all(feat == 0):
            species = idx_to_species.get(lbl)
            if species:
                feature_buffer.setdefault(species, []).append(feat)
    # One cursor, one commit for the whole chunk - see add_pokemon_features_batch.
    db.add_pokemon_features_batch(feature_buffer)


def store_features_to_db(model, dataset, db, label="checkpoint"):
    """
    Batched feature extraction + incremental DB write, using extract_batch
    (not one-image-at-a-time) so it's fast even at thousands of images.
    Called after every improved checkpoint, not just once at the very end,
    so a crash mid-training doesn't lose everything collected so far.
    """
    log.info(f"\n💾 Storing features to database ({label})...")
    t_start = time.time()
    
    image_buffer = []
    label_buffer = []
    feature_buffer: Dict[str, List[np.ndarray]] = {}
    count = 0
    
    def flush_batch():
        nonlocal count
        if not image_buffer:
            return
        pil_imgs = [transforms.ToPILImage()(img.cpu()) for img in image_buffer]
        features = model.feature_extractor.extract_batch(pil_imgs)
        for feat, lbl in zip(features, label_buffer):
            if not np.all(feat == 0):
                species = dataset.idx_to_species[lbl]
                feature_buffer.setdefault(species, []).append(feat)
                count += 1
        image_buffer.clear()
        label_buffer.clear()
    
    for img, label in dataset:
        image_buffer.append(img)
        label_buffer.append(label)
        if len(image_buffer) >= 50:
            flush_batch()
            db.add_pokemon_features_batch(feature_buffer)
            feature_buffer = {}
            gc.collect()
            _trim_memory()
    
    flush_batch()
    db.add_pokemon_features_batch(feature_buffer)
    
    elapsed = time.time() - t_start
    log.info(f"   ✅ Stored {count} features in {elapsed:.0f}s")


def _detect_container_memory_limit_mb() -> Optional[float]:
    """
    Reads the container's actual memory limit from cgroups, so the trainer
    can size itself to the real constraint instead of guessing. Tries
    cgroup v2 first (memory.max), then falls back to v1
    (memory.limit_in_bytes). Returns None if undetectable (e.g. not
    running in a container, or no limit set) rather than a fake number.
    """
    candidates = [
        "/sys/fs/cgroup/memory.max",                    # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",   # cgroup v1
    ]
    for path in candidates:
        try:
            with open(path) as f:
                raw = f.read().strip()
            if raw == "max":
                continue  # no limit set on this cgroup
            limit_bytes = int(raw)
            # cgroup v1 with no real limit often reports a huge sentinel
            # value (close to 2^63) instead of "max" - ignore those too.
            if limit_bytes <= 0 or limit_bytes > (1 << 52):
                continue
            return limit_bytes / (1024 * 1024)
        except Exception:
            continue
    return None


def _get_current_rss_mb() -> Optional[float]:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return None


def stream_train():    
    global STREAM_BATCH_SIZE
    # ============ PRE-LOAD MODEL (NO DOWNLOAD DURING TRAINING) ============

    log.info("📥 Pre-loading AI model...")
    try:
        from torchvision import models
        _ = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        log.info("✅ Model loaded and cached!")
    except Exception as e:
        log.warning(f"⚠️ Model pre-load failed: {e}")
    
    log_memory("baseline, right after model load")
    baseline_rss_mb = _get_current_rss_mb()
    container_limit_mb = _detect_container_memory_limit_mb()

    if container_limit_mb is not None:
        log.info(f"   📦 Detected container memory limit: {container_limit_mb:.0f} MB")
        if baseline_rss_mb is not None:
            # Each image in a stream batch costs roughly ~15MB of transient
            # memory during PIL conversion + transform + forward/backward
            # (measured from observed batch spikes: ~690MB baseline -> ~1150MB
            # at batch 1 with STREAM_BATCH_SIZE=32 -> (1150-690)/32 ≈ 14.4MB/img).
            # Keep total usage under 75% of the limit as a safety margin for
            # the OS, other processes, and normal fragmentation.
            safety_budget_mb = container_limit_mb * 0.75 - baseline_rss_mb
            safe_batch_size = max(4, int(safety_budget_mb / 15))
            if STREAM_BATCH_SIZE > safe_batch_size:
                log.warning(f"   ⚠️ STREAM_BATCH_SIZE={STREAM_BATCH_SIZE} looks risky for a "
                            f"{container_limit_mb:.0f}MB container with a {baseline_rss_mb:.0f}MB "
                            f"baseline - auto-lowering to {safe_batch_size} to avoid OOM. "
                            f"Set STREAM_BATCH_SIZE explicitly to override this.")
                STREAM_BATCH_SIZE = safe_batch_size
    else:
        log.info(f"   📦 Could not detect a container memory limit (not cgroup-limited, "
                 f"or running outside a container)")
    
    """Train using streaming - process in batches, clear memory."""
    
    log.info("")
    log.info("🚀 Pokémon AI Trainer - STREAMING MODE")
    log.info("=" * 60)
    log.info("   ✅ Processes images in batches")
    log.info("   ✅ Clears memory after each batch")
    log.info("   ✅ Saves to database incrementally")
    log.info("=" * 60)
    
    if HF_TOKEN:
        log.info(f"🔑 HF_TOKEN: ✅ Set")
    else:
        log.warning(f"🔑 HF_TOKEN: ❌ Not set")
    
    log.info("\n📦 Checking for archive files...")
    extract_archive_files()
    
    log.info("\n📂 Connecting to database...")
    db = Database()
    stats = db.get_stats()
    log.info(f"   Existing species: {stats['total_species']}")
    log.info(f"   Existing features: {stats['total_features']}")
    
    # Create streaming dataset
    log.info("\n📂 Initializing streaming dataset...")
    dataset = StreamingPokemonDataset(extra_dir="Extra pokemons")
    num_species = dataset.get_num_species()
    
    if num_species == 0:
        log.error("❌ No species found! Exiting.")
        return
    
    log.info(f"   📊 Total species: {num_species}")
    
    # Initialize model
    log.info("\n🧠 Initializing model...")
    model = PokemonClassifier(num_species=num_species)
    model.to(DEVICE)
    # Backbone (shufflenet) stays frozen - extract_batch always runs it under
    # torch.no_grad() regardless of this train()/eval() call. `projection`
    # has no BatchNorm/Dropout so train()/eval() doesn't change its math -
    # this call just needs to not be .eval() so nothing here is confusing
    # later. What actually makes projection trainable is passing its
    # parameters to the optimizer below + forward_batch(..., grad=True).
    model.feature_extractor.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        list(model.classifier.parameters()) + list(model.feature_extractor.projection.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    log.info(f"\n🎯 Training for {EPOCHS} epochs...")
    log.info(f"   Species: {num_species}")
    log.info(f"   Batch size: {BATCH_SIZE}")
    log.info(f"   Stream batch: {STREAM_BATCH_SIZE} images per batch")
    log.info(f"   Learning rate: {LEARNING_RATE}")
    log.info(f"   Early-stop target: {EARLY_STOP_ACC}% train acc, "
             f"or {EARLY_STOP_PATIENCE} epochs without val improvement")
    log.info("-" * 60)
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    stop_training = False
    start_epoch = 0

    # ============ RESUME FROM CHECKPOINT IF ONE EXISTS ============
    # Stored in the DB (Turso), not local disk - see Database.save_checkpoint_blob
    # for why: Railway's local container disk doesn't survive restarts here,
    # but the DB provably does (Existing species/features carry over every
    # time).
    checkpoint_blob = db.load_checkpoint_blob()
    if checkpoint_blob:
        try:
            log.info(f"\n💾 Found checkpoint in DB, resuming training...")
            checkpoint = torch.load(io.BytesIO(checkpoint_blob), map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            # A mid-epoch checkpoint (is_epoch_complete=False) means this
            # epoch's data pass never finished - re-run this SAME epoch
            # number (its weights are already partially trained, just not
            # its exact position in the data stream) instead of skipping to
            # the next one and silently dropping the rest of this epoch's
            # data forever.
            if checkpoint.get("is_epoch_complete", True):
                start_epoch = checkpoint["epoch"] + 1
            else:
                start_epoch = checkpoint["epoch"]
            best_val_acc = checkpoint.get("best_val_acc", 0.0)
            epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
            log.info(f"   ✅ Resuming at epoch {start_epoch + 1}/{EPOCHS} "
                     f"(best val acc so far: {best_val_acc:.1f}%)")
        except Exception as e:
            log.exception(f"   ❌ Failed to load checkpoint ({e}), starting fresh instead")
            start_epoch = 0
            best_val_acc = 0.0
            epochs_without_improvement = 0
    else:
        log.info(f"\n💾 No checkpoint found in DB, starting fresh")

    if start_epoch >= EPOCHS:
        log.info(f"   ℹ️ Checkpoint already completed all {EPOCHS} epochs - nothing to resume")

    # Config for periodic mid-epoch checkpointing - see note below on why
    # end-of-epoch-only checkpointing wasn't enough.
    CHECKPOINT_EVERY_N_BATCHES = int(os.getenv("CHECKPOINT_EVERY_N_BATCHES", "20"))
    last_emergency_checkpoint_batch = [-999]  # list so the batch loop can mutate it in place

    def _save_checkpoint(epoch_idx: int, is_epoch_complete: bool):
        """
        BUG FIXED: checkpointing used to happen only at the END of a
        completed epoch. With ~28,625 images to stream per epoch on this
        dataset, epoch 1 alone can take many hours - if the process gets
        killed (OOM, redeploy, crash) before finishing even ONE epoch,
        there was NO checkpoint yet to resume from at all, so a restart
        always retrained from random weights regardless of how far batch_count
        had gotten. Now this also gets called periodically mid-epoch
        (every CHECKPOINT_EVERY_N_BATCHES batches), tagged
        is_epoch_complete=False, so a mid-epoch crash still has recent
        weights to resume from - only the exact batch position within the
        epoch's data stream is lost, not the learning itself.
        """
        try:
            buf = io.BytesIO()
            torch.save({
                "epoch": epoch_idx,
                "is_epoch_complete": is_epoch_complete,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "epochs_without_improvement": epochs_without_improvement,
            }, buf)
            db.save_checkpoint_blob(buf.getvalue())
            tag = "epoch complete" if is_epoch_complete else "mid-epoch"
            log.info(f"   💾 Checkpoint saved to DB ({tag}, epoch {epoch_idx+1}/{EPOCHS}, "
                     f"{len(buf.getvalue()) / 1024:.0f} KB)")
        except Exception as e:
            # log.exception (not log.warning(str(e))) so the full traceback
            # actually shows up in the logs - a silent/vague failure here
            # was the likely reason checkpoints appeared to save but a
            # restart still found nothing to resume from.
            log.exception(f"   ❌ Failed to save checkpoint (epoch {epoch_idx+1}): {e}")

    for epoch in range(start_epoch, EPOCHS):
        log.info(f"\n📊 Epoch {epoch+1}/{EPOCHS}")
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0
        
        # Stream images
        image_buffer = []
        label_buffer = []
        
        for img, label in PrefetchIterator(dataset, maxsize=STREAM_BATCH_SIZE * 2):
            image_buffer.append(img)
            label_buffer.append(label)
            
            # When buffer reaches STREAM_BATCH_SIZE, process it
            if len(image_buffer) >= STREAM_BATCH_SIZE:
                # Create a mini-batch from buffer
                batch_imgs = torch.stack(image_buffer)
                batch_labels = torch.tensor(label_buffer)
                
                # Move to device
                batch_imgs = batch_imgs.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                
                # Forward pass
                optimizer.zero_grad()
                
                # Convert to PIL for feature extraction. Cached tensors are
                # raw uint8 (see StreamingPokemonDataset.transform) so no
                # un-normalize step is needed - ToPILImage handles uint8
                # CHW tensors directly.
                pil_imgs = []
                for i in range(batch_imgs.size(0)):
                    try:
                        pil_img = transforms.ToPILImage()(batch_imgs[i].cpu())
                        if pil_img.size[0] > 10 and pil_img.size[1] > 10:
                            pil_imgs.append(pil_img)
                    except Exception:
                        continue
                
                if not pil_imgs:
                    # Clear buffer and continue
                    image_buffer = []
                    label_buffer = []
                    continue
                
                features, logits = None, None
                try:
                    features, logits = model.forward_batch(pil_imgs, grad=True)
                    features = features.to(DEVICE)
                    logits = logits.to(DEVICE)
                    
                    loss = criterion(logits, batch_labels[:len(pil_imgs)])
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * batch_labels[:len(pil_imgs)].size(0)
                    _, predicted = torch.max(logits, 1)
                    train_total += batch_labels[:len(pil_imgs)].size(0)
                    train_correct += (predicted == batch_labels[:len(pil_imgs)]).sum().item()
                    batch_count += 1
                    
                    # Save this chunk's results to the DB right away, then
                    # it's safe to drop from memory below - nothing about
                    # this chunk needs to be revisited later.
                    store_chunk_features_to_db(features, batch_labels[:len(pil_imgs)],
                                                dataset.idx_to_species, db)
                    
                    # Progress update (every batch — cheap now that images are cached)
                    acc = 100 * train_correct / train_total if train_total > 0 else 0
                    log.info(f"   📊 Batch {batch_count}: Loss: {train_loss/train_total:.3f}, Acc: {acc:.1f}%")
                    
                    # Periodic mid-epoch checkpoint - see _save_checkpoint's
                    # docstring for why end-of-epoch-only wasn't enough.
                    if batch_count % CHECKPOINT_EVERY_N_BATCHES == 0:
                        _save_checkpoint(epoch, is_epoch_complete=False)
                    
                    # Stop this epoch early once training accuracy hits the
                    # target — no point grinding through remaining batches.
                    # Require a few batches first so one lucky batch doesn't
                    # trigger a false stop.
                    if batch_count >= 3 and acc >= EARLY_STOP_ACC:
                        log.info(f"   🎯 Reached target accuracy ({acc:.1f}% >= "
                                 f"{EARLY_STOP_ACC}%), moving on early")
                        del batch_imgs, batch_labels, pil_imgs
                        gc.collect()
                        _trim_memory()
                        break
                    
                except Exception as e:
                    log.warning(f"   ⚠️ Batch failed: {e}")
                
                # CLEAR MEMORY!
                del batch_imgs
                del batch_labels
                del pil_imgs
                del features, logits
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                _trim_memory()
                log_memory(f"batch {batch_count}")
                
                # Proactive safety net: if current usage is creeping toward
                # the container's actual limit, get a fresh checkpoint out
                # NOW rather than waiting for the next scheduled save - a
                # kernel OOM kill is a hard SIGKILL with zero warning, so
                # whatever checkpoint existed a few batches ago is all a
                # restart will have. This can't prevent the kill itself,
                # only minimize how much progress it costs.
                if container_limit_mb is not None:
                    current_rss = _get_current_rss_mb()
                    if (current_rss is not None
                            and current_rss > container_limit_mb * 0.85
                            and batch_count - last_emergency_checkpoint_batch[0] >= 5):
                        log.warning(f"   🚨 Current RSS ({current_rss:.0f} MB) is above 85% of "
                                    f"the {container_limit_mb:.0f} MB container limit - saving "
                                    f"an emergency checkpoint now")
                        _save_checkpoint(epoch, is_epoch_complete=False)
                        last_emergency_checkpoint_batch[0] = batch_count
                
                # Reset buffer
                image_buffer = []
                label_buffer = []
        
        # Process any remaining images (skip if we already hit target accuracy early)
        if image_buffer and not (batch_count >= 3 and (100 * train_correct / train_total if train_total > 0 else 0) >= EARLY_STOP_ACC):
            batch_imgs = torch.stack(image_buffer)
            batch_labels = torch.tensor(label_buffer)
            
            batch_imgs = batch_imgs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            pil_imgs = []
            for i in range(batch_imgs.size(0)):
                try:
                    pil_img = transforms.ToPILImage()(batch_imgs[i].cpu())
                    if pil_img.size[0] > 10 and pil_img.size[1] > 10:
                        pil_imgs.append(pil_img)
                except Exception:
                    continue
            
            if pil_imgs:
                try:
                    features, logits = model.forward_batch(pil_imgs, grad=True)
                    features = features.to(DEVICE)
                    logits = logits.to(DEVICE)
                    
                    loss = criterion(logits, batch_labels[:len(pil_imgs)])
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * batch_labels[:len(pil_imgs)].size(0)
                    _, predicted = torch.max(logits, 1)
                    train_total += batch_labels[:len(pil_imgs)].size(0)
                    train_correct += (predicted == batch_labels[:len(pil_imgs)]).sum().item()
                    batch_count += 1
                    
                    # Same as the main loop: persist this last chunk's
                    # results immediately rather than waiting for an
                    # end-of-epoch save.
                    store_chunk_features_to_db(features, batch_labels[:len(pil_imgs)],
                                                dataset.idx_to_species, db)
                    
                except Exception:
                    pass
                
                del batch_imgs
                del batch_labels
                del pil_imgs
                gc.collect()
                _trim_memory()
                log_memory("leftover chunk")
        
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        
        # Validation - a REAL held-out set now (see StreamingPokemonDataset:
        # VAL_IMAGES_PER_SPECIES images per species are set aside during
        # streaming and never yielded for training).
        #
        # BUG FIXED: this used to do `for img, label in dataset: ...` and
        # take the first 50 items - but iterating `dataset` from scratch
        # replays the SAME cached images in the SAME order every time,
        # which is exactly what the training loop above had just trained
        # on this epoch. Val Acc was measuring memorization of a training
        # subset, not generalization - which made "best model" checkpoint
        # selection and early stopping both unreliable.
        model.eval()
        val_correct = 0
        val_total = 0
        val_set = dataset.get_val_set()
        
        if val_set:
            with torch.no_grad():
                for i in range(0, len(val_set), STREAM_BATCH_SIZE):
                    chunk = val_set[i:i + STREAM_BATCH_SIZE]
                    batch_imgs = torch.stack([item[0] for item in chunk]).to(DEVICE)
                    batch_labels = torch.tensor([item[1] for item in chunk]).to(DEVICE)
                    
                    pil_imgs = []
                    for j in range(batch_imgs.size(0)):
                        try:
                            pil_img = transforms.ToPILImage()(batch_imgs[j].cpu())
                            if pil_img.size[0] > 10 and pil_img.size[1] > 10:
                                pil_imgs.append(pil_img)
                        except Exception:
                            continue
                    
                    if pil_imgs:
                        _, logits = model.forward_batch(pil_imgs)
                        _, predicted = torch.max(logits, 1)
                        val_total += batch_labels[:len(pil_imgs)].size(0)
                        val_correct += (predicted == batch_labels[:len(pil_imgs)]).sum().item()
                    
                    del batch_imgs, batch_labels, pil_imgs
            gc.collect()
            _trim_memory()
        else:
            log.info(f"   ℹ️ No held-out validation images yet (still on the "
                     f"first streaming pass) - Val Acc will read 0% until it "
                     f"finishes collecting")
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        log.info(f"\n📊 Epoch {epoch+1} Results:")
        log.info(f"   Train Acc: {train_acc:.1f}%")
        log.info(f"   Val Acc: {val_acc:.1f}%")
        log.info(f"   Total batches: {batch_count}")
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            os.makedirs(os.path.dirname(MODEL_OUTPUT) or ".", exist_ok=True)
            torch.save(model.feature_extractor.state_dict(), MODEL_OUTPUT)
            log.info(f"   ✅ Saved model (val acc: {val_acc:.1f}%)")
            # Write features to DB right away too — don't wait until the
            # whole run finishes. If Railway crashes on a later epoch,
            # this checkpoint's data is already safe in the DB.
            store_features_to_db(model, dataset, db, label=f"epoch {epoch+1}, val acc {val_acc:.1f}%")
        else:
            epochs_without_improvement += 1

        # Full training-state checkpoint at epoch end. Written to the DB
        # (Turso), not local disk - see Database.save_checkpoint_blob.
        _save_checkpoint(epoch, is_epoch_complete=True)

        # Stop across epochs if we've hit the target or plateaued
        if train_acc >= EARLY_STOP_ACC:
            log.info(f"   🎯 Training accuracy target reached ({train_acc:.1f}% >= "
                     f"{EARLY_STOP_ACC}%), stopping training")
            stop_training = True
        elif epochs_without_improvement >= EARLY_STOP_PATIENCE:
            log.info(f"   ⏸️ Val accuracy hasn't improved in {EARLY_STOP_PATIENCE} "
                     f"epochs (best: {best_val_acc:.1f}%), stopping early")
            stop_training = True
        
        if stop_training:
            break
    
    log.info("-" * 60)
    log.info(f"✅ Training complete!")
    log.info(f"   Best validation accuracy: {best_val_acc:.1f}%")
    log.info(f"   Model saved to: {MODEL_OUTPUT}")
    
    final_stats = db.get_stats()
    log.info(f"\n📊 Database Stats:")
    log.info(f"   Total species: {final_stats['total_species']}")
    log.info(f"   Total features: {final_stats['total_features']}")
    log.info(f"   Using Turso: {final_stats['use_turso']}")
    
    log.info("\n" + "=" * 60)
    log.info("✅ All done! Model is ready to use.")
    log.info("=" * 60)
    
    db.close()


if __name__ == "__main__":
    stream_train()
