#!/usr/bin/env python3
"""
NSFW Image Detection Script
Автоматически использует локальную папку ./models.
Выводит результат в формате JSON.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from PIL import Image
from transformers import pipeline
from tqdm import tqdm


def setup_classifier(model_path=None, retries=3):
    """Инициализация классификатора."""

    # Если указан локальный путь
    if model_path and os.path.exists(model_path):
        print(f"📂 Загрузка локальной модели из: {model_path}", file=sys.stderr)
        try:
            classifier = pipeline(
                "image-classification",
                model=model_path,
                use_fast=False
            )
            print("✅ Локальная модель загружена!\n", file=sys.stderr)
            return classifier
        except Exception as e:
            print(f"⚠️ Ошибка загрузки локальной модели: {e}", file=sys.stderr)
            sys.exit(1)

    # Загрузка из Hugging Face (Fallback)
    model_name = "Falconsai/nsfw_image_detection"
    print(f"🌐 Локальная модель не найдена. Загрузка {model_name}...", file=sys.stderr)

    for attempt in range(retries):
        try:
            classifier = pipeline(
                "image-classification",
                model=model_name
            )
            print("✅ Модель загружена успешно!\n", file=sys.stderr)
            return classifier
        except Exception as e:
            print(f"⚠️ Попытка {attempt + 1}/{retries} не удалась: {e}", file=sys.stderr)
            if attempt < retries - 1:
                import time
                time.sleep(5)
            else:
                print("\n❌ Не удалось загрузить модель. Положите файлы в папку './models'.", file=sys.stderr)
                sys.exit(1)


def classify_single_image(classifier, image_path):
    """Классификация одного изображения."""
    try:
        img = Image.open(image_path).convert("RGB")
        result = classifier(img)

        # Ищем оценку для лейбла 'nsfw'
        nsfw_score = 0.0
        for item in result:
            if item['label'] == 'nsfw':
                nsfw_score = item['score']
                break

        return {
            "path": str(image_path),
            "nsfw": nsfw_score
        }
    except Exception as e:
        return {
            "path": str(image_path),
            "error": str(e)
        }


def get_image_files(path):
    """Получение списка изображений."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}

    if os.path.isfile(path):
        if Path(path).suffix.lower() in valid_extensions:
            return [path]
        else:
            return []

    elif os.path.isdir(path):
        images = []
        for ext in valid_extensions:
            images.extend(Path(path).glob(f'*{ext}'))
            images.extend(Path(path).glob(f'*{ext.upper()}'))
        return sorted(list(set(images)))
    else:
        return []


def main():
    parser = argparse.ArgumentParser(description='NSFW Image Detection (JSON Output)')
    parser.add_argument('path', help='Путь к изображению или папке')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(json.dumps({'error': f"Путь '{args.path}' не существует"}))
        sys.exit(1)

    image_files = get_image_files(args.path)
    if not image_files:
        print(json.dumps({'error': 'Изображения не найдены'}))
        sys.exit(1)

    # Логика поиска локальной модели
    script_dir = Path(__file__).parent
    local_models_dir = script_dir / "models"

    model_path = None
    if local_models_dir.exists():
        model_path = str(local_models_dir)

    classifier = setup_classifier(model_path=model_path)

    results = []

    # Используем tqdm для прогресса в stderr, чтобы не мусорить в stdout (JSON)
    iterator = tqdm(image_files, desc="🔍 Обработка", file=sys.stderr) if len(image_files) > 1 else image_files

    for img_path in iterator:
        result = classify_single_image(classifier, img_path)
        results.append(result)

    # Вывод результата в JSON
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()