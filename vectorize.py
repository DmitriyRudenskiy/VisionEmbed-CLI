#!/usr/bin/env python3
"""
VisionEmbed-CLI: Утилита для генерации векторных эмбеддингов изображений.
Использует архитектуру Qwen-VL для извлечения признаков.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# --- Constants ---
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_RESIZE = 384
DEFAULT_OUTPUT = "embeddings.json"

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class ImageCollector:
    """Отвечает за поиск и фильтрацию изображений в директории."""

    def __init__(self, input_dir: Path):
        self.input_dir = input_dir

    def scan(self) -> list[Path]:
        """Сканирует директорию и возвращает список путей к изображениям."""
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Директория '{self.input_dir}' не существует.")

        image_files = []
        for file_path in self.input_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
                image_files.append(file_path)

        logger.info(f"Найдено файлов в директории: {len(image_files)}")
        return image_files


class ModelService:
    """Управляет загрузкой и инициализацией модели."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model: Optional[AutoModel] = None
        self.processor: Optional[AutoProcessor] = None

    def load(self) -> None:
        """Загружает модель и процессор на устройство."""
        logger.info(f"Загрузка модели из: {self.model_path} ...")
        try:
            # trust_remote_code требуется для некоторых моделей (например, Qwen)
            logger.warning("Используется trust_remote_code=True. Убедитесь в доверии к источнику модели.")

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="auto"
            )
            self.model.eval()
            logger.info("Модель успешно загружена.")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise RuntimeError(f"Failed to load model: {e}") from e


class EmbeddingProcessor:
    """Генерирует эмбеддинги для списка изображений."""

    def __init__(self, model: AutoModel, processor: AutoProcessor, resize_size: Optional[int] = None):
        self.model = model
        self.processor = processor
        self.resize_size = resize_size

    @staticmethod
    def _resize_image(image: Image.Image, max_size: int) -> Image.Image:
        """Изменяет размер изображения, сохраняя пропорции, если оно превышает max_size."""
        if max(image.size) <= max_size:
            return image

        width, height = image.size
        ratio = max_size / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def process_batch(self, image_paths: list[Path]) -> list[dict]:
        """Обрабатывает пакет изображений и возвращает список эмбеддингов."""
        results = []

        for img_path in tqdm(image_paths, desc="Обработка", unit="img"):
            try:
                image = Image.open(img_path).convert("RGB")

                if self.resize_size and self.resize_size > 0:
                    image = self._resize_image(image, self.resize_size)

                # Формирование запроса (специфично для Qwen-VL)
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": image}],
                    }
                ]

                text_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = self.processor(
                    text=[text_prompt],
                    images=[image],
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)

                    # Выбор стратегии пулинга
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embedding = outputs.pooler_output
                    else:
                        embedding = outputs.last_hidden_state.mean(dim=1)

                # Конвертация в список float
                vector = embedding.squeeze().float().cpu().numpy().tolist()

                results.append({
                    "image_path": str(img_path),
                    "embedding_dim": len(vector),
                    "embedding": vector
                })
            except Exception as e:
                logger.error(f"Ошибка обработки {img_path.name}: {e}")

        return results


class JSONStorage:
    """Управляет сохранением и загрузкой данных в JSON."""

    def __init__(self, output_path: Path):
        self.output_path = output_path

    def load_existing(self) -> list[dict]:
        """Загружает существующие данные, если файл есть."""
        if not self.output_path.exists():
            return []

        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Загружено {len(data)} существующих записей.")
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Не удалось прочитать существующий файл: {e}. Начинаем с нуля.")
            return []

    def save(self, data: list[dict]) -> None:
        """Сохраняет данные в JSON."""
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"Данные сохранены в {self.output_path}")
        except IOError as e:
            logger.error(f"Ошибка записи файла: {e}")
            raise


def get_processed_paths(existing_data: list[dict]) -> set[str]:
    """Извлекает множество уже обработанных путей из существующих данных."""
    return {item.get("image_path") for item in existing_data if "image_path" in item}


def main():
    parser = argparse.ArgumentParser(description="VisionEmbed-CLI: Векторизация изображений.")
    parser.add_argument("--input_dir", type=str, required=True, help="Путь к папке с изображениями")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Имя выходного JSON файла")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL, help="Путь к модели или ID в HF")
    parser.add_argument("--resize", type=int, default=DEFAULT_RESIZE, help="Макс. размер стороны (0 - не менять)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    # 1. Инициализация хранилища и загрузка старых данных
    storage = JSONStorage(output_path)
    existing_results = storage.load_existing()
    processed_paths = get_processed_paths(existing_results)

    # 2. Сбор изображений
    try:
        collector = ImageCollector(input_dir)
        all_image_files = collector.scan()
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    # 3. Фильтрация уже обработанных файлов
    new_image_files = [
        path for path in all_image_files
        if str(path) not in processed_paths
    ]

    if not new_image_files:
        logger.info("Все найденные изображения уже обработаны. Выход.")
        return

    logger.info(f"К обработке добавлено {len(new_image_files)} новых изображений.")

    # 4. Загрузка модели
    model_service = ModelService(args.model_path)
    try:
        model_service.load()
    except RuntimeError:
        sys.exit(1)

    # 5. Обработка
    processor = EmbeddingProcessor(
        model_service.model,
        model_service.processor,
        args.resize
    )
    new_results = processor.process_batch(new_image_files)

    # 6. Сохранение
    # Объединяем старые и новые результаты
    final_results = existing_results + new_results
    storage.save(final_results)

    logger.info(f"✅ Готово! Обработано новых: {len(new_results)}. Всего в файле: {len(final_results)}")


if __name__ == "__main__":
    main()