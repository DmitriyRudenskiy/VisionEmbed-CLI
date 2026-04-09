#!/usr/bin/env python3
"""
NSFW Image Detection Script (Final Version)
Вывод: path, nsfw_score (полная точность). Поле error только при ошибке.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from PIL import Image
from tqdm import tqdm

# Попытка импорта torch для определения устройства
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from transformers import pipeline

# Настройка логирования (в stderr)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Falconsai/nsfw_image_detection"
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
DEFAULT_BATCH_SIZE = 4


def get_image_files(path: Path) -> List[Path]:
    """Поиск изображений."""
    if path.is_file():
        if path.suffix.lower() in VALID_EXTENSIONS:
            return [path]
        return []

    if path.is_dir():
        files = []
        try:
            for item in path.iterdir():
                if item.is_file() and item.suffix.lower() in VALID_EXTENSIONS:
                    files.append(item)
        except PermissionError:
            logger.warning(f"Нет прав на чтение директории: {path}")
        return sorted(files)

    return []


def load_classifier(model_path: Optional[str], device: str, retries: int = 3) -> Any:
    """Загрузка модели."""
    classifier = None

    if model_path and Path(model_path).exists():
        logger.info(f"📂 Загрузка локальной модели: {model_path}")
        try:
            classifier = pipeline("image-classification", model=model_path, device=device)
            logger.info("✅ Локальная модель загружена.")
            return classifier
        except Exception as e:
            logger.warning(f"⚠️ Ошибка локальной модели: {e}. Пробуем HF...")

    logger.info(f"🌐 Загрузка модели {DEFAULT_MODEL_NAME}...")

    for attempt in range(retries):
        try:
            classifier = pipeline("image-classification", model=DEFAULT_MODEL_NAME, device=device)
            logger.info("✅ Модель загружена успешно.")
            return classifier
        except Exception as e:
            logger.warning(f"⚠️ Попытка {attempt + 1}/{retries} не удалась: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.critical("❌ Не удалось загрузить модель.")
                sys.exit(1)
    return classifier


def process_batch(classifier: Any, images_data: List[Tuple], threshold: float) -> List[Dict]:
    """Обработка батча изображений."""
    if not images_data:
        return []

    valid_data = [(p, img) for p, img in images_data if img is not None]
    paths = [item[0] for item in valid_data]
    pil_images = [item[1] for item in valid_data]

    results = []
    predictions = []
    error_msg = None

    if valid_data:
        try:
            predictions = classifier(pil_images)
        except Exception as e:
            logger.error(f"Ошибка инференса: {e}")
            predictions = [None] * len(valid_data)
            error_msg = str(e)

        for idx, (path, _) in enumerate(valid_data):
            if error_msg:
                results.append({
                    "path": str(path),
                    "error": f"Inference error: {error_msg}"
                })
                continue

            preds = predictions[idx]
            nsfw_score = 0.0
            if preds:
                for item in preds:
                    if item.get('label', '').lower() == 'nsfw':
                        nsfw_score = item.get('score', 0.0)
                        break

            # Выводим score без округления
            results.append({
                "path": str(path),
                "nsfw_score": nsfw_score
            })

    # Файлы с ошибкой загрузки
    for path, img in images_data:
        if img is None:
            results.append({
                "path": str(path),
                "error": "Image load error"
            })

    return results


def get_device(device_arg: str) -> str:
    """Определение устройства."""
    if device_arg != 'auto':
        return device_arg

    if not TORCH_AVAILABLE:
        return 'cpu'

    if torch.cuda.is_available():
        logger.info("🚀 CUDA detected.")
        return 'cuda'

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("🍎 MPS detected.")
        return 'mps'

    return 'cpu'


def main():
    parser = argparse.ArgumentParser(description='NSFW Detection')
    parser.add_argument('path', type=str, help='Путь к файлу или папке')
    parser.add_argument('--model', type=str, default=None, help='Путь к локальной модели')
    parser.add_argument('--device', type=str, default='auto', help='cpu, cuda, mps, auto')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Размер батча')
    parser.add_argument('--no-local', action='store_true', help='Игнорировать ./models')

    args = parser.parse_args()

    target_path = Path(args.path)
    if not target_path.exists():
        logger.error(f"Путь '{args.path}' не существует")
        print(json.dumps({"error": f"Path '{args.path}' not found"}))
        sys.exit(1)

    model_path = None
    if not args.no_local:
        if args.model:
            model_path = args.model
        else:
            local_models_dir = Path(__file__).parent / "models"
            if local_models_dir.exists():
                model_path = str(local_models_dir)

    device = get_device(args.device)
    classifier = load_classifier(model_path, device)

    image_files = get_image_files(target_path)

    if not image_files:
        logger.warning("Изображения не найдены")
        print(json.dumps({"error": "No images found"}))
        sys.exit(0)

    logger.info(f"Найдено изображений: {len(image_files)}")

    all_results = []
    current_batch = []

    progress_bar = tqdm(image_files, desc="🔍 Обработка", file=sys.stderr, unit="img")

    for file_path in progress_bar:
        try:
            img = Image.open(file_path).convert("RGB")
            current_batch.append((file_path, img))
        except Exception as e:
            logger.warning(f"⚠️ Битый файл {file_path.name}: {e}")
            all_results.append({
                "path": str(file_path),
                "error": f"Load error: {str(e)}"
            })
            continue

        if len(current_batch) >= args.batch_size:
            results = process_batch(classifier, current_batch, 0.0)  # threshold не нужен для вывода
            all_results.extend(results)
            current_batch = []

    if current_batch:
        results = process_batch(classifier, current_batch, 0.0)
        all_results.extend(results)

    # ensure_ascii=False для поддержки кириллицы в путях
    print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()