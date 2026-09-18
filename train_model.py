"""
train_model.py - Stream Processing AI Pokémon Trainer
Trains in batches, clears memory after each batch, saves to DB incrementally
"""

import os
import sys
import json
import logging
import sqlite3
import time
import zipfile
import shutil
import subprocess
import gc
import resource
import threading
import queue
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator
from io import BytesIO
import re

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
# Head training (projection + classifier on cached frozen-backbone features).
# Epochs here take seconds, not hours, so these defaults are much larger than
# the old per-image-streaming EPOCHS/LEARNING_RATE ones.
HEAD_EPOCHS = int(os.getenv("HEAD_EPOCHS", "80"))
HEAD_LR = float(os.getenv("HEAD_LR", "1e-3"))
HEAD_BATCH = int(os.getenv("HEAD_BATCH", "256"))
HEAD_WEIGHT_DECAY = float(os.getenv("HEAD_WEIGHT_DECAY", "1e-2"))
HEAD_FEATURE_DROPOUT = float(os.getenv("HEAD_FEATURE_DROPOUT", "0.2"))
HEAD_PATIENCE = int(os.getenv("HEAD_PATIENCE", "15"))  # stop if val retrieval acc doesn't improve for N epochs
# The features already in the DB were written by an untrained/inconsistent
# projection (and every epoch appended more), so they can't be compared with
# embeddings from the new model. Default: wipe and rewrite them at the end.
REPLACE_DB_FEATURES = os.getenv("REPLACE_DB_FEATURES", "true").lower() == "true"
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
    
    def clear_features(self):
        """Removes every stored feature vector + species count."""
        cursor = self._conn.cursor()
        try:
            cursor.execute("DELETE FROM pokemon_features")
            cursor.execute("DELETE FROM species_info")
            self._conn.commit()
        finally:
            try:
                cursor.close()
            except Exception:
                pass

    def bulk_add_features(self, species_to_features: Dict[str, List[np.ndarray]],
                          rows_per_stmt: int = 50, stmts_per_commit: int = 10):
        """
        Fast bulk insert: multi-row INSERT statements (50 rows each) instead of
        one round-trip per row + a COUNT(*) per species. Over a remote Turso
        connection the per-row version costs seconds per image.
        Assumes the table was just cleared (variant names restart at 1 and
        species_info.count is set, not incremented).
        """
        rows = []
        for species, feats in species_to_features.items():
            for i, feat in enumerate(feats):
                # 6 decimals is plenty for cosine similarity and ~halves the payload
                rows.append((species, f"{species}_{i + 1}", json.dumps(np.round(feat, 6).tolist())))

        cursor = self._conn.cursor()
        try:
            n_stmts = 0
            for i in range(0, len(rows), rows_per_stmt):
                chunk = rows[i:i + rows_per_stmt]
                placeholders = ",".join(["(?, ?, ?, strftime('%s', 'now'))"] * len(chunk))
                params = tuple(v for row in chunk for v in row)
                cursor.execute(
                    "INSERT OR REPLACE INTO pokemon_features "
                    "(species, variant_name, feature_vector, created_at) VALUES " + placeholders,
                    params,
                )
                n_stmts += 1
                if n_stmts % stmts_per_commit == 0:
                    self._conn.commit()
            self._conn.commit()

            items = [(sp, len(f)) for sp, f in species_to_features.items() if f]
            for i in range(0, len(items), 100):
                chunk = items[i:i + 100]
                placeholders = ",".join(["(?, ?, strftime('%s', 'now'))"] * len(chunk))
                params = tuple(v for row in chunk for v in row)
                cursor.execute(
                    "INSERT INTO species_info (species, count, last_updated) VALUES " + placeholders +
                    " ON CONFLICT(species) DO UPDATE SET count = excluded.count, "
                    "last_updated = excluded.last_updated",
                    params,
                )
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

BACKBONE_DIM = 1280  # EfficientNet-B0's pooled feature size


class PokemonFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        # EfficientNet-B0 (ImageNet ~77.7% top-1, 5.3M params). Much stronger
        # features than the ShuffleNetV2 x0.5 it replaced (~60.6%), at ~10x the
        # CPU cost per image - fine here because the backbone is frozen and
        # embedded once per image (see build_feature_bank), and at inference
        # it's still well under a second per image on CPU.
        # torchvision's forward() does features -> avgpool -> flatten ->
        # classifier, so swapping the classifier for Identity yields a flat
        # (N, 1280) vector.
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        classifier = self.backbone.classifier
        if isinstance(classifier, nn.Sequential):
            linear_layers = [layer for layer in classifier if isinstance(layer, nn.Linear)]
            if not linear_layers:
                raise RuntimeError(f"Could not find a Linear layer in EfficientNet classifier: {classifier!r}")
            backbone_dim = linear_layers[-1].in_features
        elif isinstance(classifier, nn.Linear):
            backbone_dim = classifier.in_features
        else:
            raise RuntimeError(f"Unsupported EfficientNet classifier type: {type(classifier).__name__}")
        self.backbone.classifier = nn.Identity()
        
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


class CosineHead(nn.Module):
    """
    Cosine-softmax classifier. The embeddings are L2-normalised (unit norm),
    and the old Linear->ReLU->Dropout->Linear head on unit-norm inputs
    produced near-zero logits, so the loss sat at ln(num_classes) for
    ages. Scaling the cosine similarities fixes that and trains exactly the
    geometry the bot uses at lookup time (cosine similarity to stored
    features). Only the feature extractor is saved/deployed, so this head
    never has to match anything outside training.
    """
    def __init__(self, in_dim: int, num_classes: int, scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))


class PokemonClassifier(nn.Module):
    def __init__(self, num_species: int):
        super().__init__()
        self.feature_extractor = PokemonFeatureExtractor()
        self.classifier = CosineHead(256, num_species)
    
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

            with open(os.devnull, "w") as devnull:
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
            # Cap total species to MAX_SPECIES, prioritizing local ones
            local_list = sorted(local_species)  # deterministic order
            hf_list = sorted(s for s in all_species if s not in local_species)

            # First take up to MAX_SPECIES local species
            selected = local_list[:MAX_SPECIES]

            # Then fill remaining slots from HF, if any
            if len(selected) < MAX_SPECIES:
                remaining = MAX_SPECIES - len(selected)
                selected += hf_list[:remaining]

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

@torch.no_grad()
def _backbone_features(fe: "PokemonFeatureExtractor", batch_u8: torch.Tensor) -> torch.Tensor:
    """
    uint8 (B,3,224,224) -> frozen EfficientNet-B0 features (B,1280).
    Same math as PokemonFeatureExtractor.extract_batch (÷255 -> Normalize ->
    backbone), minus the pointless tensor->PIL->tensor round trip.
    """
    x = fe.normalize(batch_u8.float().div_(255.0))
    return fe.backbone(x)


@torch.no_grad()
def build_feature_bank(items, fe: "PokemonFeatureExtractor", batch_size: int,
                       total_hint: int = 0, tag: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One pass over `items` (any iterable of (uint8 image tensor, label)),
    returning (features [N,1280], labels [N]).

    The backbone is frozen and the cached images are already augmented, so
    its output for a given image never changes between epochs - computing it
    once and training the head on the cached vectors is mathematically the
    same as the old per-epoch forward pass, ~1000x cheaper, and lets us
    shuffle freely (see train_head).
    """
    feats: List[torch.Tensor] = []
    labels: List[int] = []
    buf_i: List[torch.Tensor] = []
    buf_l: List[int] = []
    n = 0
    n_flush = 0
    t0 = time.time()

    def flush():
        nonlocal n, n_flush
        if not buf_i:
            return
        feats.append(_backbone_features(fe, torch.stack(buf_i)).cpu())
        labels.extend(buf_l)
        n += len(buf_i)
        n_flush += 1
        buf_i.clear()
        buf_l.clear()
        if n_flush % 25 == 0:
            of = f"/{total_hint}" if total_hint else ""
            log.info(f"   🧊 [{tag}] {n}{of} images embedded ({n / max(time.time() - t0, 1e-6):.0f} img/s)")
        if n_flush % 100 == 0:
            gc.collect()
            _trim_memory()
            log_memory(f"{tag} feature pass")

    for img, label in items:
        buf_i.append(img)
        buf_l.append(int(label))
        if len(buf_i) >= batch_size:
            flush()
    flush()

    if not feats:
        return torch.empty(0, BACKBONE_DIM), torch.empty(0, dtype=torch.long)
    return torch.cat(feats), torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def _embed(fe: "PokemonFeatureExtractor", X: torch.Tensor, bs: int = 2048) -> torch.Tensor:
    out = [F.normalize(fe.projection(X[i:i + bs]), p=2, dim=1) for i in range(0, X.size(0), bs)]
    return torch.cat(out) if out else torch.empty(0, 256)


@torch.no_grad()
def _evaluate(model: "PokemonClassifier", Etr, ytr, Xv, yv) -> Tuple[float, float]:
    """
    Returns (classifier top-1 %, retrieval 1-NN %) on the held-out set.
    Retrieval = nearest stored training embedding by cosine similarity, which
    is how the bot actually uses the DB, so that's what we select on.
    """
    if Xv.size(0) == 0:
        return 0.0, 0.0
    Ev = _embed(model.feature_extractor, Xv)
    cls_acc = (model.classifier(Ev).argmax(1) == yv).float().mean().item() * 100
    correct = 0
    for i in range(0, Ev.size(0), 512):
        sims = Ev[i:i + 512] @ Etr.T
        correct += (ytr[sims.argmax(1)] == yv[i:i + 512]).sum().item()
    return cls_acc, 100 * correct / Ev.size(0)


def train_head(model: "PokemonClassifier", Xtr, ytr, Xv, yv) -> float:
    """
    Trains projection + classifier on cached backbone features with proper
    shuffling. The old loop fed the head images in the cache's on-disk order
    (grouped species by species), so every 20-image batch was ~1 class and
    the head could only chase "whatever class is current" - loss stayed at
    ln(1119) ≈ 7.0 forever no matter what the model or LR was.
    Leaves the BEST epoch's weights loaded in `model`; returns its val
    retrieval accuracy.
    """
    fe = model.feature_extractor
    params = list(fe.projection.parameters()) + list(model.classifier.parameters())
    optimizer = optim.AdamW(params, lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(HEAD_EPOCHS, 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    N = Xtr.size(0)
    has_val = Xv.size(0) > 0
    if not has_val:
        log.warning("   ⚠️ No held-out validation set - training all epochs and keeping the last one")

    best_acc, best_epoch, best_state, stale = -1.0, 0, None, 0

    for epoch in range(HEAD_EPOCHS):
        t0 = time.time()
        # projection + classifier have no BatchNorm; the (frozen) backbone
        # never runs here, so nothing can drift.
        fe.projection.train()
        model.classifier.train()

        perm = torch.randperm(N)
        loss_sum, correct = 0.0, 0
        for i in range(0, N, HEAD_BATCH):
            idx = perm[i:i + HEAD_BATCH]
            xb = F.dropout(Xtr[idx], p=HEAD_FEATURE_DROPOUT, training=True)
            yb = yb = ytr[idx]
            emb = F.normalize(fe.projection(xb), p=2, dim=1)
            logits = model.classifier(emb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
        scheduler.step()

        fe.projection.eval()
        model.classifier.eval()
        Etr = _embed(fe, Xtr)
        cls_acc, ret_acc = _evaluate(model, Etr, ytr, Xv, yv)
        log.info(f"   📊 Epoch {epoch + 1}/{HEAD_EPOCHS}: loss {loss_sum / N:.3f}, "
                 f"train acc {100 * correct / N:.1f}%, val acc {cls_acc:.1f}%, "
                 f"val retrieval {ret_acc:.1f}% ({time.time() - t0:.1f}s)")

        score = ret_acc if has_val else float(epoch)
        if score > best_acc:
            best_acc, best_epoch, stale = score, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if has_val and stale >= HEAD_PATIENCE:
                log.info(f"   ⏸️ No val improvement in {HEAD_PATIENCE} epochs "
                         f"(best: {best_acc:.1f}% at epoch {best_epoch}), stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return best_acc if has_val else 0.0


def write_features_to_db(model: "PokemonClassifier", Xtr, ytr, dataset, db):
    """
    Re-embeds every training image with the FINAL (best) model and writes
    the vectors to the DB, so the DB always matches the saved model.
    """
    log.info("\n💾 Writing features to database...")
    t0 = time.time()
    E = _embed(model.feature_extractor, Xtr).numpy()
    per_species: Dict[str, List[np.ndarray]] = {}
    for emb, lbl in zip(E, ytr.tolist()):
        species = dataset.idx_to_species.get(lbl)
        if species:
            per_species.setdefault(species, []).append(emb)
    if REPLACE_DB_FEATURES:
        log.info("   🧹 Clearing old features first (REPLACE_DB_FEATURES=true)")
        db.clear_features()
    db.bulk_add_features(per_species)
    log.info(f"   ✅ Stored {len(E)} features for {len(per_species)} species in {time.time() - t0:.0f}s")


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
        _ = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
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
    log.info("   ✅ Streams/caches images once, embeds with the frozen backbone")
    log.info("   ✅ Trains the head on shuffled cached features")
    log.info("   ✅ Writes features matching the final model to the database")
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
    # The EfficientNet backbone is FROZEN, so it must stay in eval() for the
    # whole run. The old code called feature_extractor.train(), which put the
    # backbone's BatchNorm layers in TRAIN mode: every (single-species)
    # batch was normalised with its own statistics and overwrote the
    # pretrained running mean/var - so the features drifted, and the saved
    # model file contained corrupted BN buffers. Nothing below calls .train()
    # on the backbone; train_head only toggles projection/classifier.
    model.eval()
    for p in model.feature_extractor.backbone.parameters():
        p.requires_grad_(False)

    # ============ PHASE 1: FROZEN-BACKBONE FEATURES (one pass) ============
    log.info("\n🧊 Phase 1/3: embedding every image once with the frozen backbone...")
    t0 = time.time()
    fe = model.feature_extractor
    total_hint = num_species * max(MAX_IMAGES_PER_SPECIES - VAL_IMAGES_PER_SPECIES, 0)
    # PrefetchIterator overlaps Hugging Face streaming / disk reads with compute.
    Xtr, ytr = build_feature_bank(PrefetchIterator(dataset, maxsize=STREAM_BATCH_SIZE * 2),
                                  fe, STREAM_BATCH_SIZE, total_hint=total_hint, tag="train")
    # The held-out set is only complete once the pass above has finished.
    Xv, yv = build_feature_bank(dataset.get_val_set(), fe, STREAM_BATCH_SIZE, tag="val")
    gc.collect()
    _trim_memory()
    log.info(f"   ✅ {Xtr.size(0)} train + {Xv.size(0)} val images embedded in {time.time() - t0:.0f}s")
    log_memory("after feature pass")

    if Xtr.size(0) == 0:
        log.error("❌ No training images were collected! Exiting.")
        db.close()
        return
    seen = int(ytr.unique().numel())
    if seen < num_species:
        log.warning(f"   ⚠️ Only {seen}/{num_species} species have training images")

    # ============ PHASE 2: TRAIN PROJECTION + CLASSIFIER ============
    log.info(f"\n🎯 Phase 2/3: training head ({Xtr.size(0)} samples, {num_species} species, "
             f"{HEAD_EPOCHS} epochs max, lr {HEAD_LR}, batch {HEAD_BATCH})")
    log.info("-" * 60)
    t0 = time.time()
    best_acc = train_head(model, Xtr, ytr, Xv, yv)
    log.info("-" * 60)
    log.info(f"   ✅ Head trained in {time.time() - t0:.0f}s (best val retrieval acc: {best_acc:.1f}%)")

    os.makedirs(os.path.dirname(MODEL_OUTPUT) or ".", exist_ok=True)
    torch.save(model.feature_extractor.state_dict(), MODEL_OUTPUT)
    log.info(f"   ✅ Saved model to {MODEL_OUTPUT}")

    # ============ PHASE 3: WRITE FEATURES TO DB ============
    log.info("\n🗄️ Phase 3/3: database")
    write_features_to_db(model, Xtr, ytr, dataset, db)

    log.info("-" * 60)
    log.info(f"✅ Training complete!")
    log.info(f"   Best validation retrieval accuracy: {best_acc:.1f}%")
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
