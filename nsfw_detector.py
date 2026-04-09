#!/usr/bin/env python3
"""
NSFW Image Detection Script (Optimized & Refactored)
Использование: python script.py <path>
"""

import json
import logging
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Generator,
    List,
    Optional,
    Protocol,
    Tuple,  # <--- Исправлено: добавлен Tuple
    TypedDict,
    Union,
    cast,
    runtime_checkable,
)

from PIL import Image
from tqdm import tqdm

# --- Configuration ---
DEFAULT_MODEL = "Falconsai/nsfw_image_detection"
BATCH_SIZE = 8
VALID_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'})

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Graceful Shutdown ---
_shutdown_requested = False


def signal_handler(signum: int, frame) -> None:
    global _shutdown_requested
    if not _shutdown_requested:
        logger.warning("⚠️ Получен сигнал завершения. Сохраняем результаты...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# --- Type Definitions ---

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


@runtime_checkable
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


# --- Helpers ---

def get_device() -> str:
    """Определение устройства для инференса."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


def find_local_model() -> Optional[Path]:
    """Поиск локальной модели."""
    local_path = Path(__file__).parent / "models"
    if (local_path / "config.json").exists():
        return local_path
    return None


def load_model() -> ClassifierProtocol:
    """Загрузка модели с ретраями."""
    from transformers import pipeline

    model_path = find_local_model()
    model_id = str(model_path) if model_path else DEFAULT_MODEL
    device = get_device()

    src = "📂 Локальная" if model_path else "🌐 HuggingFace"
    logger.info(f"{src} модель: {model_id} | Device: {device.upper()}")

    last_exc = None
    for attempt in range(3):
        try:
            return pipeline(
                "image-classification",
                model=model_id,
                device=device,
                batch_size=BATCH_SIZE
            )
        except Exception as e:
            last_exc = e
            logger.warning(f"Попытка {attempt + 1}/3 не удалась: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Не удалось загрузить модель: {last_exc}")


def scan_files(path: Path) -> List[Path]:
    """Сканирование директории или файла."""
    if path.is_file():
        return [path] if path.suffix.lower() in VALID_EXTENSIONS else []

    if path.is_dir():
        files = [p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
        files.sort()
        return files

    raise FileNotFoundError(f"Путь не найден: {path}")


@contextmanager
def json_array_output():
    """Контекстный менеджер для потокового вывода JSON массива."""
    sys.stdout.write('[\n')
    separator = ""
    try:
        def write_item(data: dict):
            nonlocal separator
            sys.stdout.write(f"{separator}{json.dumps(data, ensure_ascii=False)}")
            separator = ",\n"

        yield write_item
    finally:
        sys.stdout.write('\n]\n')
        sys.stdout.flush()


# --- Core Logic ---

def process_batch(
        classifier: ClassifierProtocol,
        batch: List[Tuple[Path, Image.Image]]
) -> List[ProcessingResult]:
    """Обработка батча изображений."""
    paths, images = zip(*batch)  # Unzip

    try:
        predictions = classifier(list(images))

        results = []
        for path, preds in zip(paths, predictions):
            nsfw_score = 0.0
            for pred in preds:
                if pred['label'].lower() == 'nsfw':
                    nsfw_score = pred['score']
                    break
            results.append(ProcessingResult(path=path, nsfw_score=float(nsfw_score)))
        return results

    except Exception as e:
        error_msg = f"Inference error: {e}"
        logger.error(f"Ошибка инференса: {e}")
        return [ProcessingResult(path=p, error=error_msg) for p in paths]


def main():
    if len(sys.argv) != 2:
        print(f"Использование: {sys.argv[0]} <path>", file=sys.stderr)
        sys.exit(1)

    target_path = Path(sys.argv[1])

    # 1. Инициализация
    try:
        classifier = load_model()
    except Exception as e:
        logger.critical(f"Ошибка инициализации модели: {e}")
        sys.exit(1)

    # 2. Сканирование
    try:
        files = scan_files(target_path)
    except Exception as e:
        logger.error(f"Ошибка сканирования: {e}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    if not files:
        logger.warning("Изображения не найдены")
        print(json.dumps([]))
        sys.exit(0)

    logger.info(f"Найдено файлов: {len(files)}")

    # 3. Обработка
    processed_count = 0
    error_count = 0
    batch: List[Tuple[Path, Image.Image]] = []

    with json_array_output() as write_item, \
            tqdm(total=len(files), desc="🔍 Обработка", file=sys.stderr, unit="img") as pbar:

        for file_path in files:
            if _shutdown_requested:
                break

            try:
                img = Image.open(file_path).convert("RGB")
                batch.append((file_path, img))
            except Exception as e:
                error_count += 1
                write_item(ProcessingResult(path=file_path, error=f"Load error: {e}").to_dict())
                pbar.update(1)
                continue

            if len(batch) >= BATCH_SIZE:
                results = process_batch(classifier, batch)

                for res in results:
                    write_item(res.to_dict())
                    if res.error:
                        error_count += 1

                processed_count += len(results)
                pbar.update(len(batch))
                batch.clear()

        if batch and not _shutdown_requested:
            results = process_batch(classifier, batch)
            for res in results:
                write_item(res.to_dict())
                if res.error:
                    error_count += 1
            processed_count += len(results)
            pbar.update(len(batch))

    logger.info(f"✅ Готово. Обработано: {processed_count}, Ошибок: {error_count}")


if __name__ == "__main__":
    main()