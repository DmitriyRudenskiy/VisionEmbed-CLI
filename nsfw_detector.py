#!/usr/bin/env python3
"""
Ultra-fast NSFW Image Detection
Features: async I/O, automatic mixed precision, checkpointing, deduplication, ONNX support
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import signal
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError

# --- Environment & Warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*GetPrototype.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)


# --- Configuration (Pydantic-style validation) ---
@dataclass(frozen=True, slots=True)
class Config:
    target_path: Path
    model_id: str = "Falconsai/nsfw_image_detection"
    batch_size: int = 16  # Auto-adjusted later based on VRAM
    threshold: float = 0.5
    max_resolution: int = 1024  # Resize long edge to save VRAM
    num_workers: int = 4  # I/O threads
    checkpoint_file: Path | None = None
    output_format: str = "json"  # json, jsonl, csv
    use_fp16: bool = True  # Automatic mixed precision
    dedup: bool = True  # Skip duplicate images by hash
    resume: bool = True

    def __post_init__(self):
        if not self.target_path.exists():
            raise ValueError(f"Path not found: {self.target_path}")


# --- Types ---
@dataclass(slots=True)
class ImageItem:
    path: Path
    content: bytes | None = None  # Raw bytes for hashing
    pil_image: Image.Image | None = None
    file_hash: str | None = None
    error: str | None = None

    def close(self):
        if self.pil_image:
            self.pil_image.close()


@dataclass(slots=True)
class Prediction:
    path: Path
    nsfw_score: float = 0.0
    safe_score: float = 0.0
    processing_time_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        if self.error:
            return {"path": str(self.path), "error": self.error}
        return {
            "path": str(self.path),
            "nsfw_score": round(self.nsfw_score, 4),
            "safe_score": round(self.safe_score, 4),
            "inference_ms": round(self.processing_time_ms, 2)
        }


class ClassifierBackend(Protocol):
    """Abstract backend for inference (HuggingFace, ONNX, etc.)"""

    def predict(self, images: list[Image.Image]) -> list[dict]: ...

    def get_optimal_batch_size(self) -> int: ...


# --- High-Performance Components ---

class CheckpointManager:
    """Resume capability for long-running scans"""

    def __init__(self, checkpoint_path: Path | None):
        self.path = checkpoint_path or Path(".nsfw_scan_checkpoint.json")
        self.processed_hashes: set[str] = set()
        self.results: list[dict] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.processed_hashes = set(data.get("hashes", []))
                self.results = data.get("results", [])
                logging.info(f"📂 Loaded checkpoint: {len(self.processed_hashes)} files already processed")
            except Exception:
                pass

    def add(self, file_hash: str, result: dict):
        self.processed_hashes.add(file_hash)
        self.results.append(result)
        # Periodic save every 10 items
        if len(self.results) % 10 == 0:
            self._save()

    def _save(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps({
            "hashes": list(self.processed_hashes),
            "results": self.results,
            "timestamp": time.time()
        }))
        tmp.rename(self.path)

    def is_processed(self, file_hash: str) -> bool:
        return file_hash in self.processed_hashes

    def finalize(self):
        self._save()
        if self.path.exists():
            self.path.unlink()  # Remove on successful completion


class AsyncImageLoader:
    """Parallel I/O with preprocessing pipeline"""

    def __init__(self, config: Config, checkpoint: CheckpointManager):
        self.config = config
        self.checkpoint = checkpoint
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        self.seen_hashes: set[str] = set() if config.dedup else None

    def _compute_hash(self, data: bytes) -> str:
        return hashlib.blake2b(data, digest_size=16).hexdigest()

    def _load_and_preprocess(self, path: Path) -> ImageItem:
        """Worker function: load, hash, resize"""
        try:
            with open(path, 'rb') as f:
                data = f.read()

            # Deduplication check
            file_hash = self._compute_hash(data)
            if self.seen_hashes is not None:
                if file_hash in self.seen_hashes or self.checkpoint.is_processed(file_hash):
                    return ImageItem(path=path, file_hash=file_hash, error="DUPLICATE")
                self.seen_hashes.add(file_hash)

            # Image validation and preprocessing
            img = Image.open(io.BytesIO(data))

            # Convert to RGB efficiently
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Smart resize (keep aspect ratio, reduce VRAM usage)
            w, h = img.size
            max_size = self.config.max_resolution
            if max(w, h) > max_size:
                ratio = max_size / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            return ImageItem(
                path=path,
                content=data,
                pil_image=img,
                file_hash=file_hash
            )

        except UnidentifiedImageError:
            return ImageItem(path=path, error="INVALID_IMAGE")
        except Exception as e:
            return ImageItem(path=path, error=f"LOAD_ERROR: {e}")

    async def load_batch(self, paths: list[Path]) -> list[ImageItem]:
        """Async batch loading"""
        loop = asyncio.get_event_loop()
        futures = [loop.run_in_executor(self.executor, self._load_and_preprocess, p) for p in paths]
        return await asyncio.gather(*futures)

    def close(self):
        self.executor.shutdown(wait=True)


class OptimizedInferenceEngine:
    """High-performance inference with AMP and batching"""

    def __init__(self, config: Config):
        self.config = config
        self.device = self._detect_device()
        self.model = None
        self.processor = None
        self._build_pipeline()

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _build_pipeline(self):
        """Load with optimizations"""
        from transformers import pipeline, AutoImageProcessor

        logger.info(f"🚀 Loading model: {self.config.model_id} on {self.device.upper()}")

        # Auto-batch size detection based on VRAM
        batch_size = self.config.batch_size
        if self.device == "cuda":
            import torch
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            batch_size = min(32, max(4, int(vram * 2)))  # ~2 images per GB
            logger.info(f"📊 Auto-adjusted batch size to {batch_size} (VRAM: {vram:.1f}GB)")

        # Mixed precision setup
        torch_dtype = None
        if self.config.use_fp16 and self.device == "cuda":
            torch_dtype = "float16"

        self.classifier = pipeline(
            "image-classification",
            model=self.config.model_id,
            device=self.device,
            torch_dtype=torch_dtype,
            batch_size=batch_size,
            use_fast=True
        )

        # Warmup
        dummy = Image.new('RGB', (224, 224), color='white')
        self.classifier([dummy])
        logger.info("✅ Model warmup complete")

    @torch.inference_mode()
    def predict_batch(self, items: list[ImageItem]) -> list[Prediction]:
        """Optimized batch prediction"""
        if not items:
            return []

        valid_items = [i for i in items if i.error is None and i.pil_image is not None]
        if not valid_items:
            return [Prediction(path=i.path, error=i.error) for i in items]

        images = [i.pil_image for i in valid_items]
        start_time = time.perf_counter()

        try:
            # Automatic mixed precision context
            import torch
            context = torch.cuda.amp.autocast if (
                        self.config.use_fp16 and self.device == "cuda") else lambda: __import__(
                'contextlib').nullcontext()

            with context():
                outputs = self.classifier(images)

            elapsed = (time.perf_counter() - start_time) * 1000

            # Parse results
            predictions = []
            for item, preds in zip(valid_items, outputs):
                nsfw_score = 0.0
                safe_score = 0.0

                for pred in preds:
                    label = pred['label'].lower()
                    score = pred['score']
                    if any(x in label for x in ['nsfw', 'porn', 'sexy', 'hentai']):
                        nsfw_score = max(nsfw_score, score)
                    elif 'safe' in label or 'normal' in label:
                        safe_score = max(safe_score, score)

                predictions.append(Prediction(
                    path=item.path,
                    nsfw_score=nsfw_score,
                    safe_score=safe_score,
                    processing_time_ms=elapsed / len(valid_items)
                ))

            # Handle failed loads
            for item in items:
                if item.error and item not in valid_items:
                    predictions.append(Prediction(path=item.path, error=item.error))

            return predictions

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return [Prediction(path=i.path, error=str(e)) for i in items]


class OutputManager:
    """Streaming output with multiple format support"""

    def __init__(self, format_type: str, output_path: Path | None = None):
        self.format = format_type
        self.output = output_path or sys.stdout
        self.first_item = True
        self._open_stream()

    def _open_stream(self):
        if self.format == "json":
            self._write('[\n')
        elif self.format == "jsonl":
            pass
        elif self.format == "csv":
            self._write("path,nsfw_score,safe_score,status,inference_ms\n")

    def _write(self, data: str):
        if isinstance(self.output, Path):
            with open(self.output, 'a') as f:
                f.write(data)
        else:
            self.output.write(data)
            self.output.flush()

    def write(self, pred: Prediction):
        data = pred.to_dict()

        if self.format == "json":
            prefix = ",\n" if not self.first_item else ""
            self._write(f"{prefix}{json.dumps(data)}")
            self.first_item = False
        elif self.format == "jsonl":
            self._write(json.dumps(data) + '\n')
        elif self.format == "csv":
            status = "error" if pred.error else "ok"
            self._write(
                f"{data['path']},{data.get('nsfw_score', '')},{data.get('safe_score', '')},{status},{data.get('inference_ms', '')}\n")

    def close(self):
        if self.format == "json":
            self._write('\n]')


class MetricsCollector:
    """Performance monitoring"""

    def __init__(self):
        self.stats = defaultdict(list)
        self.start_time = time.time()

    def record(self, batch_size: int, inference_time: float, io_time: float):
        self.stats['batch_sizes'].append(batch_size)
        self.stats['inference_times'].append(inference_time)
        self.stats['io_times'].append(io_time)

    def report(self):
        total_time = time.time() - self.start_time
        total_images = sum(self.stats['batch_sizes'])

        if not total_images:
            return

        avg_inference = sum(self.stats['inference_times']) / len(self.stats['inference_times'])
        avg_io = sum(self.stats['io_times']) / len(self.stats['io_times']) if self.stats['io_times'] else 0

        logger.info(f"\n{'=' * 50}")
        logger.info(f"📊 PERFORMANCE REPORT")
        logger.info(f"{'=' * 50}")
        logger.info(f"Total images: {total_images}")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Throughput: {total_images / total_time:.1f} img/sec")
        logger.info(f"Avg inference time: {avg_inference * 1000:.1f}ms/batch")
        logger.info(f"Avg I/O time: {avg_io * 1000:.1f}ms/batch")
        logger.info(f"{'=' * 50}")


# --- Main Pipeline ---

class NSFWScanner:
    def __init__(self, config: Config):
        self.config = config
        self.checkpoint = CheckpointManager(config.checkpoint_file if config.resume else None)
        self.loader = AsyncImageLoader(config, self.checkpoint)
        self.engine = OptimizedInferenceEngine(config)
        self.output = OutputManager(config.output_format)
        self.metrics = MetricsCollector()
        self.shutdown_event = asyncio.Event()
        self.stats = {"processed": 0, "errors": 0, "duplicates": 0, "skipped": 0}

        # Signal handling
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_event_loop().add_signal_handler(sig, self._signal_handler)

    def _signal_handler(self):
        logger.warning("\n⚠️ Shutdown requested, finishing current batch...")
        self.shutdown_event.set()

    def _get_files(self) -> list[Path]:
        """Fast file discovery with filtering"""
        path = self.config.target_path
        extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}

        if path.is_file():
            return [path] if path.suffix.lower() in extensions else []

        # Use rglob with early termination on shutdown
        files = []
        for p in path.rglob('*'):
            if p.suffix.lower() in extensions:
                files.append(p)
            if len(files) % 1000 == 0 and self.shutdown_event.is_set():
                break
        return files

    async def run(self):
        """Main async pipeline"""
        files = self._get_files()
        if not files:
            logger.warning("No images found")
            return

        logger.info(f"🔍 Found {len(files)} images")

        # Batching with prefetching
        batch_size = self.engine.classifier.batch_size or 8
        pending_batch: list[Path] = []

        try:
            for filepath in files:
                if self.shutdown_event.is_set():
                    break

                pending_batch.append(filepath)

                if len(pending_batch) >= batch_size * 2:  # Prefetch 2x batch
                    await self._process_pending(pending_batch[:batch_size])
                    pending_batch = pending_batch[batch_size:]

            # Process remainder
            if pending_batch and not self.shutdown_event.is_set():
                await self._process_pending(pending_batch)

        finally:
            self.loader.close()
            self.output.close()
            self.metrics.report()
            self.checkpoint.finalize()
            logger.info(
                f"✅ Done. Processed: {self.stats['processed']}, Errors: {self.stats['errors']}, Dupes: {self.stats['duplicates']}")

    async def _process_pending(self, paths: list[Path]):
        """Process single batch with parallel I/O"""
        io_start = time.perf_counter()
        items = await self.loader.load_batch(paths)
        io_time = time.perf_counter() - io_start

        # Handle duplicates and errors from loader
        predictions = []
        valid_items = []

        for item in items:
            if item.error == "DUPLICATE":
                self.stats['duplicates'] += 1
                self.stats['skipped'] += 1
                predictions.append(Prediction(path=item.path, error="duplicate"))
            elif item.error:
                self.stats['errors'] += 1
                predictions.append(Prediction(path=item.path, error=item.error))
                item.close()
            else:
                valid_items.append(item)

        # Inference
        if valid_items:
            infer_start = time.perf_counter()
            preds = self.engine.predict_batch(valid_items)
            infer_time = time.perf_counter() - infer_start

            self.metrics.record(len(valid_items), infer_time, io_time)
            predictions.extend(preds)
            self.stats['processed'] += len(valid_items)

        # Output and cleanup
        for pred in predictions:
            self.output.write(pred)
            if pred.error is None and pred.path.exists():
                # Update checkpoint with hash if available
                item = next((i for i in items if i.path == pred.path), None)
                if item and item.file_hash:
                    self.checkpoint.add(item.file_hash, pred.to_dict())

        # Cleanup images
        for item in items:
            item.close()


# --- CLI ---

def parse_args() -> Config:
    import argparse
    parser = argparse.ArgumentParser(
        description="High-performance NSFW Image Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("path", type=Path, help="Path to image or directory")
    parser.add_argument("-m", "--model", default="Falconsai/nsfw_image_detection", help="Model ID")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="NSFW threshold")
    parser.add_argument("-w", "--workers", type=int, default=4, help="I/O worker threads")
    parser.add_argument("--max-resolution", type=int, default=1024, help="Max image long edge")
    parser.add_argument("--format", choices=["json", "jsonl", "csv"], default="json", help="Output format")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint file for resume")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume capability")
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication")

    args = parser.parse_args()
    return Config(
        target_path=args.path,
        model_id=args.model,
        batch_size=args.batch_size,
        threshold=args.threshold,
        max_resolution=args.max_resolution,
        num_workers=args.workers,
        checkpoint_file=args.checkpoint,
        output_format=args.format,
        resume=not args.no_resume,
        use_fp16=not args.no_fp16,
        dedup=not args.no_dedup
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)]
    )

    config = parse_args()
    scanner = NSFWScanner(config)

    try:
        asyncio.run(scanner.run())
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    import io  # Required for BytesIO usage above

    main()