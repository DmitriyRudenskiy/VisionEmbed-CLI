import os
import sys
import shutil
import re
import argparse
from PIL import Image


def process_images(source_dir, recursive=False):
    # Проверяем, существует ли директория
    if not os.path.exists(source_dir):
        print(f"Ошибка: Директория '{source_dir}' не найдена.")
        return

    # Список поддерживаемых расшираний
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.avif')

    # Папка для оригиналов
    backup_dir_name = "___!ORIGINAL_IMAGE"

    # Регулярное выражение для определения уже обработанных файлов
    processed_pattern = re.compile(r"^\d+_\d{7}\.jpg$", re.IGNORECASE)

    files_data = []

    print(f"Сканирование файлов (Рекурсивно: {recursive})...")

    # Логика сканирования зависит от флага recursive
    if recursive:
        # os.walk проходит по всем вложенным папкам
        for root, dirs, files in os.walk(source_dir):
            # Исключаем папку с бэкапами из обхода, чтобы не обрабатывать их повторно
            if backup_dir_name in dirs:
                dirs.remove(backup_dir_name)

            for filename in files:
                full_path = os.path.join(root, filename)

                # Пропускаем файлы, которые уже были обработаны
                if processed_pattern.match(filename):
                    continue

                if filename.lower().endswith(valid_extensions):
                    try:
                        creation_time = os.path.getctime(full_path)
                        # Добавляем root, чтобы знать, в какой папке находится файл
                        files_data.append((creation_time, full_path, filename, root))
                    except OSError as e:
                        print(f"Не удалось прочитать дату создания файла {filename}: {e}")
    else:
        # Стандартный нерекурсивный обход (как было изначально)
        for filename in os.listdir(source_dir):
            full_path = os.path.join(source_dir, filename)

            if os.path.isdir(full_path):
                continue

            if processed_pattern.match(filename):
                continue

            if filename.lower().endswith(valid_extensions):
                try:
                    creation_time = os.path.getctime(full_path)
                    # Передаем source_dir как папку назначения
                    files_data.append((creation_time, full_path, filename, source_dir))
                except OSError as e:
                    print(f"Не удалось прочитать дату создания файла {filename}: {e}")

    if not files_data:
        print("Нет новых изображений для обработки.")
        return

    # Сортировка от ранних к поздним
    files_data.sort(key=lambda x: x[0])

    print(f"Найдено {len(files_data)} изображений. Начинаю обработку...")

    for index, (ctime, filepath, old_name, file_dir) in enumerate(files_data):
        try:
            # Формируем Unix Timestamp
            timestamp_str = str(int(ctime))

            # Порядковый номер из 7 цифр
            serial_number = f"{index + 1:07d}"

            # Новое имя файла
            new_filename = f"{timestamp_str}_{serial_number}.jpg"

            # Путь для нового файла (в той же директории, где лежит оригинал)
            output_path = os.path.join(file_dir, new_filename)

            # Папка для оригиналов (создается в той же директории, где лежит оригинал)
            backup_dir_path = os.path.join(file_dir, backup_dir_name)

            # Создаем папку для оригиналов, если её нет
            if not os.path.exists(backup_dir_path):
                os.makedirs(backup_dir_path)
                # print(f"Создана папка для оригиналов: {backup_dir_path}")

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
                base, ext = os.path.splitext(old_name)
                counter = 1
                while os.path.exists(os.path.join(backup_dir_path, f"{base}_{counter}{ext}")):
                    counter += 1
                original_move_path = os.path.join(backup_dir_path, f"{base}_{counter}{ext}")

            shutil.move(filepath, original_move_path)
            # print(f"[ПЕРЕМЕЩЕН] {old_name} -> {backup_dir_name}/")

        except Exception as e:
            print(f"[ОШИБКА] Не удалось обработать {old_name}: {e}")

    print("\nГотово!")


if __name__ == "__main__":
    # Настройка парсера аргументов
    parser = argparse.ArgumentParser(description="Переименование и конвертация изображений.")

    parser.add_argument(
        "path",
        help="Путь к папке с изображениями"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Включить рекурсивный обход вложенных папок (по умолчанию выключен)"
    )

    args = parser.parse_args()

    process_images(args.path, args.recursive)