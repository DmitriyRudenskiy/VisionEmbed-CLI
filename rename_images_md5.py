import hashlib
import os
import sys


def get_md5(filepath):
    """Вычисляет MD5 хеш файла."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def process_directory(directory):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    try:
        # ИСПРАВЛЕНИЕ 1: Добавлено условие not f.startswith('.')
        # Это проигнорирует файлы ._*.jpg, .DS_Store и другие системные файлы
        files = [
            f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and not f.startswith('.')
        ]
    except FileNotFoundError:
        print(f"Ошибка: Директория '{directory}' не найдена.")
        return

    for filename in files:
        old_path = os.path.join(directory, filename)
        name_part, ext = os.path.splitext(filename)

        if ext.lower() not in valid_extensions:
            continue

        try:
            file_hash = get_md5(old_path)
        except FileNotFoundError:
            # ИСПРАВЛЕНИЕ 2: Обработка ситуации, когда файл исчез прямо во время работы скрипта
            print(f"Файл не найден (возможно, был удален): {filename}")
            continue
        except PermissionError:
            print(f"Нет доступа к файлу: {filename}")
            continue

        # Стандартное новое имя: хеш + расширение
        new_filename = f"{file_hash}{ext}"
        new_path = os.path.join(directory, new_filename)

        if filename == new_filename:
            continue

        # Проверка на существование файла
        if os.path.exists(new_path):
            # Добавляем постфикс _duplicate перед расширением
            base_new_name = f"{file_hash}_duplicate"
            new_filename = f"{base_new_name}{ext}"
            new_path = os.path.join(directory, new_filename)

            # Если и такой файл существует, добавляем счетчик
            counter = 1
            while os.path.exists(new_path):
                new_filename = f"{base_new_name}_{counter}{ext}"
                new_path = os.path.join(directory, new_filename)
                counter += 1

        try:
            os.rename(old_path, new_path)
            print(f"Переименовано: {filename} -> {new_filename}")
        except OSError as e:
            print(f"Ошибка при переименовании {filename}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python rename_images.py <путь_к_папке>")
    else:
        target_dir = sys.argv[1]
        process_directory(target_dir)