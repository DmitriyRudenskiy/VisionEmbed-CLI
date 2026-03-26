import os
import sys
import shutil
import re
from PIL import Image


def process_images(source_dir):
    # Проверяем, существует ли директория
    if not os.path.exists(source_dir):
        print(f"Ошибка: Директория '{source_dir}' не найдена.")
        return

    # Список поддерживаемых расшираний
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.avif')

    # Папка для оригиналов
    backup_dir_name = "___!ORIGINAL_IMAGE"
    backup_dir_path = os.path.join(source_dir, backup_dir_name)

    # Регулярное выражение для определения уже обработанных файлов
    # Формат: числа_7цифр.jpg (например, 1698765432_0000001.jpg)
    processed_pattern = re.compile(r"^\d+_\d{7}\.jpg$", re.IGNORECASE)

    files_data = []

    print("Сканирование файлов...")
    for filename in os.listdir(source_dir):
        full_path = os.path.join(source_dir, filename)

        # Пропускаем папки (в том числе нашу папку для оригиналов)
        if os.path.isdir(full_path):
            continue

        # Пропускаем файлы, которые уже были обработаны этим скриптом
        if processed_pattern.match(filename):
            continue

        if filename.lower().endswith(valid_extensions):
            try:
                creation_time = os.path.getctime(full_path)
                files_data.append((creation_time, full_path, filename))
            except OSError as e:
                print(f"Не удалось прочитать дату создания файла {filename}: {e}")

    if not files_data:
        print("Нет новых изображений для обработки.")
        return

    # Сортировка от ранних к поздним
    files_data.sort(key=lambda x: x[0])

    # Создаем папку для оригиналов, если её нет
    if not os.path.exists(backup_dir_path):
        os.makedirs(backup_dir_path)
        print(f"Создана папка для оригиналов: {backup_dir_path}")

    print(f"Найдено {len(files_data)} изображений. Начинаю обработку...")

    for index, (ctime, filepath, old_name) in enumerate(files_data):
        try:
            # Формируем Unix Timestamp
            timestamp_str = str(int(ctime))

            # Порядковый номер из 7 цифр
            serial_number = f"{index + 1:07d}"

            # Новое имя файла
            new_filename = f"{timestamp_str}_{serial_number}.jpg"

            # Путь для нового файла (в той же директории, что и оригинал)
            output_path = os.path.join(source_dir, new_filename)

            # Конвертация и сохранение
            with Image.open(filepath) as img:
                # Конвертируем в RGB
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")

                # Сохраняем новый файл
                img.save(output_path, "JPEG", quality=100)

            print(f"[СОЗДАН] {new_filename}")

            # Перемещение оригинала
            original_move_path = os.path.join(backup_dir_path, old_name)

            # Проверка на случай, если файл с таким именем уже есть в папке оригиналов
            if os.path.exists(original_move_path):
                # Если вдруг совпало имя, добавляем суффикс, чтобы не потерять файл
                base, ext = os.path.splitext(old_name)
                counter = 1
                while os.path.exists(os.path.join(backup_dir_path, f"{base}_{counter}{ext}")):
                    counter += 1
                original_move_path = os.path.join(backup_dir_path, f"{base}_{counter}{ext}")

            shutil.move(filepath, original_move_path)
            print(f"[ПЕРЕМЕЩЕН] {old_name} -> {backup_dir_name}/")

        except Exception as e:
            print(f"[ОШИБКА] Не удалось обработать {old_name}: {e}")

    print("\nГотово!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python rename_images.py <путь_к_папке>")
        print("Пример: python rename_images.py C:\\Users\\User\\Pictures")
    else:
        input_dir = sys.argv[1]
        process_images(input_dir)