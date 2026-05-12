import argparse
import json
import os
from PIL import Image
from dwpose import DwposeDetector

# Допустимые расширения изображений
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')


def pad_to_square(img, target_size=1024):
    """Вписывает изображение в квадрат target_size x target_size с сохранением пропорций (черный фон)."""
    # Конвертируем в RGB, чтобы избежать проблем с альфа-каналом (RGBA) при создании черного фона
    img = img.convert("RGB")

    old_width, old_height = img.size
    ratio = min(target_size / old_width, target_size / old_height)
    new_width = int(old_width * ratio)
    new_height = int(old_height * ratio)

    # Масштабируем с хорошим качеством
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Создаем черный квадрат
    new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))

    # Вычисляем отступы, чтобы разместить картинку по центру
    pad_x = (target_size - new_width) // 2
    pad_y = (target_size - new_height) // 2

    # Вставляем масштабированное изображение в центр черного квадрата
    new_img.paste(img, (pad_x, pad_y))
    return new_img


def process_image(image_path, model, prefix, target_size=1024):
    """Обработка одного изображения."""
    try:
        print(f"Обработка: {image_path}")
        img = Image.open(image_path)

        # Вписываем в квадрат 1024x1024
        img_squared = pad_to_square(img, target_size=target_size)

        # Запуск детектора (detect_resolution теперь 1024)
        imgOut, j, source = model(
            img_squared,
            include_hand=True,
            include_face=True,
            include_body=True,
            image_and_json=True,
            detect_resolution=target_size  # Устанавливаем разрешение анализа равным размеру квадрата
        )

        # Формирование путей для сохранения
        dir_name = os.path.dirname(image_path)
        file_name, ext = os.path.splitext(os.path.basename(image_path))

        # Базовое имя с префиксом
        out_base_name = f"{prefix}{file_name}"
        out_base_path = os.path.join(dir_name, out_base_name)

        # Сохранение JSON
        json_path = f"{out_base_path}.json"
        with open(json_path, "w") as f:
            json.dump(j, f)

        # Сохранение OpenPose изображения (оно тоже будет 1024x1024 с паддингом)
        openpose_path = f"{out_base_path}_openpose{ext}"
        imgOut.save(openpose_path)

        # Сохранение исходного изображения (обрезанного до detect_resolution)
        source_path = f"{out_base_path}_source{ext}"
        source.save(source_path)

        print(f"Успешно сохранено: {out_base_name}_*")

    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="DWPose консольный обработчик изображений.")
    parser.add_argument("input", type=str, help="Путь к картинке или директории с изображениями")
    parser.add_argument("--prefix", type=str, default="dwpose_",
                        help="Префикс для сохраняемых файлов (по умолчанию: dwpose_)")
    parser.add_argument("--size", type=int, default=1024, help="Размер квадрата для вписывания (по умолчанию: 1024)")

    args = parser.parse_args()
    input_path = args.input
    prefix = args.prefix
    target_size = args.size

    if not os.path.exists(input_path):
        print(f"Ошибка: Путь '{input_path}' не существует.")
        return

    # Инициализация модели (один раз)
    print("Загрузка модели DWPose...")
    model = DwposeDetector.from_pretrained_default()
    print("Модель загружена!")

    if os.path.isfile(input_path):
        # Если передан конкретный файл
        process_image(input_path, model, prefix, target_size)
    elif os.path.isdir(input_path):
        # Если передана директория
        print(f"Поиск изображений в директории: {input_path}")
        files = os.listdir(input_path)
        image_files = [f for f in files if f.lower().endswith(VALID_EXTENSIONS)]

        if not image_files:
            print("В директории не найдено изображений с подходящим расширением.")
            return

        for filename in image_files:
            file_path = os.path.join(input_path, filename)
            process_image(file_path, model, prefix, target_size)

        print("\nОбработка директории завершена!")
    else:
        print("Ошибка: Переданный путь не является ни файлом, ни директорией.")


if __name__ == "__main__":
    main()