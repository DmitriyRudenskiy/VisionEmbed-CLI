#!/usr/bin/env python3
"""
NSFW Image Detection Script (Refactored & Optimized)
Usage: python script.py <path>
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Protocol, runtime_checkable

from PIL import Image

# --- Suppress Known Warnings ---
warnings.filterwarnings("ignore", message=".*GetPrototype.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
logging.getLogger("protobuf").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- Configuration Defaults ---
DEFAULT_MODEL = "Falconsai/nsfw_image_detection"
DEFAULT_BATCH_SIZE = 8
DEFAULT_THRESHOLD = 0.0
VALID_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'})

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# --- Types & Data Structures ---

@dataclass(frozen=True)
class Config:
    target_path: Path
    model_id: str = DEFAULT_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    threshold: float = DEFAULT_THRESHOLD


@runtime_checkable
class ClassifierProtocol(Protocol):
    def __call__(self, images: list[Image.Image], **kwargs) -> list[list[dict]]: ...


@dataclass(slots=True)
class ScanResult:
    path: Path
    nsfw_score: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        if self.error:
            return {"path": str(self.path), "error": self.error}
        return {"path": str(self.path), "nsfw_score": self.nsfw_score}


# --- Global State for Graceful Shutdown ---
class ShutdownManager:
    def __init__(self):
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum: int, frame) -> None:
        if not self._shutdown_requested:
            logger.warning("⚠️ Получен сигнал завершения. Завершаем текущий батч...")
        self._shutdown_requested = True

    @property
    def requested(self) -> bool:
        return self._shutdown_requested


shutdown = ShutdownManager()


# --- Helpers ---

def get_device() -> str:
    """Определение устройства для инференса с кэшированием."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


def scan_files(path: Path) -> Generator[Path, None, None]:
    """Генератор для сканирования файлов (экономия памяти)."""
    if path.is_file():
        if path.suffix.lower() in VALID_EXTENSIONS:
            yield path
        return

    if path.is_dir():
        for p in path.rglob('*'):
            if shutdown.requested:
                break
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                yield p
    else:
        raise FileNotFoundError(f"Путь не найден: {path}")


@contextmanager
def json_array_output(output_stream=None):
    """Контекстный менеджер для валидного потокового JSON массива."""
    stream = output_stream or sys.stdout
    stream.write('[\n')
    first = True
    try:
        def write_item(data: dict):
            nonlocal first
            if not first:
                stream.write(',\n')
            stream.write(json.dumps(data, ensure_ascii=False))
            first = False
            stream.flush()

        yield write_item
    finally:
        stream.write('\n]')
        stream.flush()


# --- Core Components ---

class ModelManager:
    def __init__(self, model_id: str, batch_size: int):
        self.model_id = model_id
        self.batch_size = batch_size
        self.classifier: Optional[ClassifierProtocol] = None
        self.device = get_device()

    def load(self) -> ClassifierProtocol:
        """Загрузка модели с ретраями."""
        from transformers import pipeline

        local_path = Path(__file__).parent / "models"
        model_src = str(local_path) if (local_path / "config.json").exists() else self.model_id
        src_label = "📂 Локальная" if (local_path / "config.json").exists() else "🌐 HuggingFace"

        logger.info(f"{src_label} модель: {model_src} | Device: {self.device.upper()}")

        last_exc = None
        for attempt in range(3):
            try:
                self.classifier = pipeline(
                    "image-classification",
                    model=model_src,
                    device=self.device,
                    batch_size=self.batch_size
                )
                return self.classifier
            except Exception as e:
                last_exc = e
                logger.warning(f"Попытка {attempt + 1}/3 не удалась: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"Не удалось загрузить модель: {last_exc}")


class ImageProcessor:
    def __init__(self, classifier: ClassifierProtocol, threshold: float):
        self.classifier = classifier
        self.threshold = threshold

    def process_batch(self, batch: list[tuple[Path, Image.Image]]) -> list[ScanResult]:
        """Обработка батча изображений с освобождением ресурсов."""
        paths = [item[0] for item in batch]
        images = [item[1] for item in batch]
        results = []

        try:
            predictions = self.classifier(images)

            for path, preds in zip(paths, predictions):
                score = 0.0
                for pred in preds:
                    label = pred.get('label', '').lower()
                    if 'nsfw' in label or 'porn' in label or 'sexy' in label:
                        score = pred.get('score', 0.0)
                        break

                results.append(ScanResult(path=path, nsfw_score=float(score)))

        except Exception as e:
            logger.error(f"Ошибка инференса батча: {e}", exc_info=True)
            error_msg = f"Inference error: {e}"
            results = [ScanResult(path=p, error=error_msg) for p in paths]

        finally:
            for _, img in batch:
                try:
                    img.close()
                except Exception:
                    pass

        return results


# --- Main Logic ---

def parse_args() -> Config:
    import argparse
    parser = argparse.ArgumentParser(description="NSFW Image Detection")
    parser.add_argument("path", type=Path, help="Path to file or directory")
    args = parser.parse_args()
    return Config(target_path=args.path)


def main():
    config = parse_args()

    # 1. Init Model
    try:
        model_mgr = ModelManager(config.model_id, config.batch_size)
        classifier = model_mgr.load()
    except Exception as e:
        logger.critical(f"Ошибка инициализации модели: {e}")
        sys.exit(1)

    # 2. Scan Files
    try:
        files_list = list(scan_files(config.target_path))
    except Exception as e:
        logger.error(f"Ошибка сканирования: {e}")
        sys.exit(1)

    if not files_list:
        logger.warning("Изображения не найдены")
        sys.stdout.write('[]')
        sys.stdout.flush()
        sys.exit(0)

    logger.info(f"Найдено файлов: {len(files_list)}")

    # 3. Process
    processor = ImageProcessor(classifier, config.threshold)
    batch: list[tuple[Path, Image.Image]] = []
    processed_count = 0
    error_count = 0

    with json_array_output() as write_item:
        for file_path in files_list:
            if shutdown.requested:
                break

            try:
                img = Image.open(file_path).convert("RGB")
                batch.append((file_path, img))
            except Exception as e:
                error_count += 1
                write_item(ScanResult(path=file_path, error=f"Load error: {e}").to_dict())
                processed_count += 1
                continue

            if len(batch) >= config.batch_size:
                results = processor.process_batch(batch)
                for res in results:
                    write_item(res.to_dict())
                    if res.error:
                        error_count += 1
                processed_count += len(results)
                batch.clear()

        if batch and not shutdown.requested:
            results = processor.process_batch(batch)
            for res in results:
                write_item(res.to_dict())
                if res.error:
                    error_count += 1
            processed_count += len(results)

    logger.info(f"✅ Готово. Обработано: {processed_count}, Ошибок: {error_count}")


if __name__ == "__main__":
    main()