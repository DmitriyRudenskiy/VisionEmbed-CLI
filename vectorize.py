import argparse
import json
import os
import sys
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm

# Константы
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


class ImageCollector:
    """Отвечает за поиск и сканирование директории с изображениями."""

    def __init__(self, input_dir: str):
        self.input_dir = input_dir

    def scan(self) -> list[str]:
        if not os.path.isdir(self.input_dir):
            raise FileNotFoundError(f"Директория '{self.input_dir}' не существует.")

        image_files = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if Path(file).suffix.lower() in ALLOWED_EXTENSIONS:
                    image_files.append(os.path.join(root, file))
        return image_files


class ModelService:
    """Отвечает за загрузку модели и процессора."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None

    def load(self):
        print(f"🚀 Загрузка модели из: {self.model_path} ...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, device_map="auto")
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {e}")


class EmbeddingProcessor:
    """Отвечает за генерацию эмбеддингов для списка изображений."""

    def __init__(self, model: AutoModel, processor: AutoProcessor, resize_size: int = None):
        self.model = model
        self.processor = processor
        self.resize_size = resize_size

    @staticmethod
    def _resize_image(image: Image.Image, max_size: int):
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

    def process_batch(self, image_paths: list[str]) -> list[dict]:
        results = []

        for img_path in tqdm(image_paths, desc="Обработка", unit="img"):
            try:
                image = Image.open(img_path).convert("RGB")

                if self.resize_size and self.resize_size > 0:
                    image = self._resize_image(image, self.resize_size)

                # Формирование запроса для Qwen-VL (без явного текста)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                        ],
                    }
                ]

                text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)

                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embedding = outputs.pooler_output
                    else:
                        embedding = outputs.last_hidden_state.mean(dim=1)

                # Исправление ошибки BFloat16: явное приведение к float32
                vector = embedding.squeeze().float().cpu().numpy().tolist()

                results.append({
                    "image_path": img_path,
                    "embedding_dim": len(vector),
                    "embedding": vector
                })
            except Exception as e:
                tqdm.write(f"❌ Ошибка {os.path.basename(img_path)}: {e}")

        return results


class JSONStorage:
    """Отвечает за сохранение и чтение результатов в JSON."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def load_existing(self) -> list[dict]:
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def save(self, data: list[dict]):
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="VisionEmbed-CLI: Векторизация изображений.")
    parser.add_argument("--input_dir", type=str, required=True, help="Путь к папке с изображениями")
    parser.add_argument("--output", type=str, default="embeddings.json", help="Имя выходного JSON файла")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-Embedding-2B", help="Путь к модели")
    parser.add_argument("--resize", type=int, default=384, help="Макс. размер стороны (0 - не менять)")
    args = parser.parse_args()

    # 1. Сбор изображений (Сначала ищем файлы)
    try:
        collector = ImageCollector(args.input_dir)
        image_files = collector.scan()
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

    if not image_files:
        print("⚠️ В указанной директории не найдено поддерживаемых изображений. Завершение работы.")
        return

    print(f"📂 Найдено изображений: {len(image_files)}.")

    # 2. Загрузка модели (Только если изображения есть)
    model_service = ModelService(args.model_path)
    try:
        model_service.load()
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

    # 3. Загрузка старых данных (если есть)
    storage = JSONStorage(args.output)
    results = storage.load_existing()

    # 4. Обработка
    processor = EmbeddingProcessor(model_service.model, model_service.processor, args.resize)
    new_results = processor.process_batch(image_files)

    # Добавляем новые результаты к старым (или перезаписываем логику можно изменить тут)
    results.extend(new_results)

    # 5. Сохранение
    storage.save(results)

    print(f"\n✅ Готово! Всего обработано: {len(new_results)}. Результат в: {args.output}")


if __name__ == "__main__":
    main()