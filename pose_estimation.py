import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoConfig, AutoModel


def process_image(image_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    input_path = Path(image_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    output_path = input_path.with_name(f"{input_path.stem}_result.json")

    print("Загрузка модели rtmw-x-384x288...")
    config = AutoConfig.from_pretrained("akore/rtmw-x-384x288", trust_remote_code=True)
    model = AutoModel.from_pretrained("akore/rtmw-x-384x288", trust_remote_code=True)
    model.to(device)
    model.eval()

    print(f"Обработка изображения: {input_path.name}")
    image = Image.open(input_path).convert("RGB")

    # РУЧНАЯ ПРЕДОБРАБОТКА (вместо багичного AutoImageProcessor)
    # Стандартные параметры нормализации для моделей MMPose (ImageNet)
    transform = transforms.Compose([
        transforms.Resize((288, 384)),  # Высота 288, Ширина 384
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pixel_values = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # coordinate_mode="model" вернет координаты в пространстве 288x384
        outputs = model(pixel_values=pixel_values, coordinate_mode="model")

    keypoints = outputs.keypoints[0].cpu().tolist()
    scores = outputs.scores[0].cpu().tolist()

    # Заменяем NaN на 0.0 для чистоты JSON
    scores = [0.0 if s != s else s for s in scores]

    result_data = {
        "image": input_path.name,
        "keypoints": keypoints,
        "scores": scores
    }

    print(f"Сохранение результата в: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)

    print("Готово!")


def main():
    parser = argparse.ArgumentParser(description="Оценка позы с помощью rtmw-x.")
    parser.add_argument("image", type=str, help="Путь к изображению")
    args = parser.parse_args()
    process_image(args.image)


if __name__ == "__main__":
    main()