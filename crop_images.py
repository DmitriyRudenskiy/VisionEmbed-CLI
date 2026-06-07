import argparse
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError


def crop_center_square(img):
    """Вырезает центральный квадрат из изображения."""
    width, height = img.size
    min_dim = min(width, height)

    # Вычисляем координаты для центрального кропа
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2

    return img.crop((left, top, right, bottom))


def process_directory(directory_path):
    dir_path = Path(directory_path)

    if not dir_path.is_dir():
        print(f"Ошибка: Директория '{directory_path}' не существует.")
        return

    # Поддерживаемые расширения
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

    processed_count = 0

    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            try:
                # Открываем изображение и переворачиваем согласно EXIF (чтобы не было перевернутых фото с телефона)
                with Image.open(file_path) as img:
                    img = img.convert("RGB")  # Защита от RGBA при сохранении в JPG
                    cropped_img = crop_center_square(img)

                    # Формируем новое имя: оригинал_crop.ext
                    new_name = f"{file_path.stem}_crop{file_path.suffix}"
                    new_path = file_path.with_name(new_name)

                    # Сохраняем с оптимальным качеством
                    cropped_img.save(new_path, quality=95)
                    processed_count += 1
                    print(f"Обработано: {file_path.name} -> {new_name}")

            except UnidentifiedImageError:
                print(f"Пропущено (не изображение): {file_path.name}")
            except Exception as e:
                print(f"Ошибка при обработке {file_path.name}: {e}")

    print(f"\nГотово! Обработано изображений: {processed_count}")


if __name__ == "__main__":
    # Настройка парсера аргументов командной строки
    parser = argparse.ArgumentParser(
        description="Вырезание центрального квадрата из изображений в указанной директории.")
    parser.add_argument("directory", help="Путь к директории с изображениями")

    args = parser.parse_args()
    process_directory(args.directory)