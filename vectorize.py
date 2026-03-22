import argparse
import json
import os
import sys
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm  # Библиотека для прогресс-бара

# Расширения файлов, которые будем обрабатывать
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def resize_image(image, max_size=512):
    """Изменяет размер изображения, сохраняя пропорции."""
    if max(image.size) <= max_size:
        return image

    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser(
        description="VisionEmbed-CLI: Векторизация изображений в JSON с использованием VL-моделей.",
        epilog="Пример: python vectorize.py --input_dir ./images --model_path ./my_model"
    )

    parser.add_argument("--input_dir", type=str, required=True, help="Путь к папке с изображениями")
    parser.add_argument("--output", type=str, default="embeddings.json", help="Имя выходного JSON файла")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-Embedding-2B",
                        help="Путь к модели (локальная папка или ID HuggingFace)")
    parser.add_argument("--resize", type=int, default=384, help="Макс. размер стороны изображения (0 - не менять)")

    args = parser.parse_args()

    # Проверки и загрузка
    if not os.path.isdir(args.input_dir):
        print(f"❌ Ошибка: Директория '{args.input_dir}' не существует.")
        sys.exit(1)

    print(f"🚀 Загрузка модели из: {args.model_path} ...")
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, device_map="auto")
        model.eval()
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        sys.exit(1)

    # Сбор файлов
    image_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if Path(file).suffix.lower() in ALLOWED_EXTENSIONS:
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("⚠️ В указанной директории не найдено поддерживаемых изображений.")
        return

    print(f"📂 Найдено изображений: {len(image_files)}. Начинаю векторизацию...\n")

    results = []
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
        except:
            pass

    resize_val = args.resize if args.resize > 0 else None

    # Обработка с прогресс-баром
    for img_path in tqdm(image_files, desc="Обработка", unit="img"):
        try:
            image = Image.open(img_path).convert("RGB")
            if resize_val:
                image = resize_image(image, resize_val)

            inputs = processor(images=image, text="", return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output
                else:
                    embedding = outputs.last_hidden_state.mean(dim=1)

            vector = embedding.squeeze().cpu().numpy().tolist()

            results.append({
                "image_path": img_path,
                "embedding_dim": len(vector),
                "embedding": vector
            })
        except Exception as e:
            tqdm.write(f"❌ Ошибка {os.path.basename(img_path)}: {e}")

    # Сохранение
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Готово! Всего обработано: {len(image_files)}. Результат в: {args.output}")


if __name__ == "__main__":
    main()