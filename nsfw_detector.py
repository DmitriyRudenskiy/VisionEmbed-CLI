#!/usr/bin/env python3
"""
NSFW Image Detection Script (Simplified)
Использование: python script.py <path>
"""

import gc
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Protocol, Tuple, TypedDict, Union, cast

from PIL import Image
from tqdm import tqdm


# Type definitions
class PredictionItem(TypedDict):
    label: str
    score: float


class SuccessResult(TypedDict):
    path: str
    nsfw_score: float


class ErrorResult(TypedDict):
    path: str
    error: str


ResultType = Union[SuccessResult, ErrorResult]


class ClassifierProtocol(Protocol):
    def __call__(self, images: List[Image.Image]) -> List[List[PredictionItem]]: ...


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    path: Path
    nsfw_score: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> ResultType:
        if self.error:
            return ErrorResult(path=str(self.path), error=self.error)
        return SuccessResult(path=str(self.path), nsfw_score=cast(float, self.nsfw_score))


# Constants
DEFAULT_MODEL = "Falconsai/nsfw_image_detection"
BATCH_SIZE = 4
VALID_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'})

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Graceful shutdown
_shutdown_requested = False


def signal_handler(signum: int, frame) -> None:
    global _shutdown_requested
    logger.warning(f"\n⚠️  Завершение работы...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_device() -> str:
    """Автоматическое определение устройства"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("🚀 CUDA")
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("🍎 MPS")
            return 'mps'
    except ImportError:
        pass
    logger.info("💻 CPU")
    return 'cpu'


def find_local_model() -> Optional[Path]:
    """Поиск локальной модели в ./models относительно скрипта"""
    script_dir = Path(__file__).parent
    local_path = script_dir / "models"
    if local_path.exists() and (local_path / "config.json").exists():
        return local_path
    return None


def load_model() -> ClassifierProtocol:
    """Загрузка модели (локальной или из HF)"""
    from transformers import pipeline

    local_model = find_local_model()
    model_id = str(local_model) if local_model else DEFAULT_MODEL

    if local_model:
        logger.info(f"📂 Локальная модель: {local_model}")
    else:
        logger.info(f"🌐 Загрузка из HF: {DEFAULT_MODEL}")

    device = get_device()

    for attempt in range(3):
        try:
            classifier = pipeline(
                "image-classification",
                model=model_id,
                device=device,
                batch_size=BATCH_SIZE
            )
            logger.info("✅ Модель загружена")
            return cast(ClassifierProtocol, classifier)
        except Exception as e:
            logger.warning(f"⚠️ Попытка {attempt + 1}/3: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    raise RuntimeError("Не удалось загрузить модель")


def get_images(path: Path) -> Generator[Path, None, None]:
    """Поиск изображений (файл или папка)"""
    if path.is_file():
        if path.suffix.lower() in VALID_EXTENSIONS:
            yield path
        else:
            logger.warning(f"Неподдерживаемый формат: {path.suffix}")
    elif path.is_dir():
        try:
            for item in sorted(path.iterdir()):
                if item.is_file() and item.suffix.lower() in VALID_EXTENSIONS:
                    yield item
        except PermissionError:
            logger.error(f"Нет доступа к директории: {path}")
    else:
        raise FileNotFoundError(f"Путь не найден: {path}")


def process_batch(classifier: ClassifierProtocol, batch: List[Tuple[Path, Image.Image]]) -> Generator[
    ProcessingResult, None, None]:
    """Обработка батча"""
    if not batch:
        return

    paths = [p for p, _ in batch]
    images = [img for _, img in batch]

    try:
        predictions = classifier(images)

        for path, preds in zip(paths, predictions):
            if _shutdown_requested:
                break

            nsfw_score = 0.0
            for pred in preds:
                if pred.get('label', '').lower() == 'nsfw':
                    nsfw_score = pred.get('score', 0.0)
                    break

            yield ProcessingResult(path=path, nsfw_score=float(nsfw_score))

    except Exception as e:
        logger.error(f"Ошибка инференса: {e}")
        for path in paths:
            yield ProcessingResult(path=path, error=f"Inference error: {e}")

    # Очистка памяти
    del images
    gc.collect()


def main():
    if len(sys.argv) != 2:
        print(f"Использование: {sys.argv[0]} <path>", file=sys.stderr)
        sys.exit(1)

    target_path = Path(sys.argv[1])

    # Загрузка модели
    try:
        classifier = load_model()
    except Exception as e:
        logger.critical(f"Ошибка инициализации: {e}")
        sys.exit(1)

    # Подсчет файлов для прогресса
    try:
        files = list(get_images(target_path))
        total = len(files)
        file_gen = iter(files)
    except Exception as e:
        logger.error(f"Ошибка сканирования: {e}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    if not files:
        logger.warning("Изображения не найдены")
        print(json.dumps([]))
        sys.exit(0)

    logger.info(f"Найдено: {total} изображений")

    # Потоковая обработка и вывод
    sys.stdout.write('[\n')
    first = True

    batch: List[Tuple[Path, Image.Image]] = []
    processed = errors = 0

    with tqdm(total=total, desc="🔍 Обработка", file=sys.stderr, unit="img") as pbar:
        for file_path in file_gen:
            if _shutdown_requested:
                break

            # Загрузка
            try:
                with Image.open(file_path) as img:
                    batch.append((file_path, img.convert("RGB")))
            except Exception as e:
                errors += 1
                result = ProcessingResult(path=file_path, error=f"Load error: {e}")
                if not first:
                    sys.stdout.write(',\n')
                json.dump(result.to_dict(), sys.stdout, ensure_ascii=False)
                first = False
                pbar.update(1)
                continue

            # Обработка батча
            if len(batch) >= BATCH_SIZE:
                for result in process_batch(classifier, batch):
                    if not first:
                        sys.stdout.write(',\n')
                    json.dump(result.to_dict(), sys.stdout, ensure_ascii=False)
                    first = False
                    processed += 1
                    if result.error:
                        errors += 1
                pbar.update(len(batch))
                batch = []

        # Остаток
        if batch and not _shutdown_requested:
            for result in process_batch(classifier, batch):
                if not first:
                    sys.stdout.write(',\n')
                json.dump(result.to_dict(), sys.stdout, ensure_ascii=False)
                first = False
                processed += 1
                if result.error:
                    errors += 1
            pbar.update(len(batch))

    sys.stdout.write('\n]\n')
    sys.stdout.flush()

    logger.info(f"✅ Готово: {processed} обработано, {errors} ошибок")


if __name__ == "__main__":
    main()