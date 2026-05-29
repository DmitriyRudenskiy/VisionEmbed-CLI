import argparse
from pathlib import Path
import shutil


def move_files_to_root(target_dir: Path):
    # Разрешаем полный путь для избежания ошибок при смене рабочих директорий
    root_dir = target_dir.resolve()

    if not root_dir.is_dir():
        print(f"Ошибка: '{root_dir}' не является директорией или не существует.")
        return

    print(f"Начало обработки директории: {root_dir}")

    # Рекурсивно обходим все файлы в директории и поддиректориях
    for file_path in root_dir.rglob('*'):
        # Нас интересуют только файлы, которые находятся НЕ в корневой директории
        if file_path.is_file() and file_path.parent != root_dir:

            # Формируем целевой путь файла в корневой директории
            target_path = root_dir / file_path.name

            # Проверка на конфликт имен
            if target_path.exists():
                # Если файл с таким именем уже есть в корне, добавляем счетчик
                counter = 1
                stem = file_path.stem  # имя файла без расширения
                suffix = file_path.suffix  # расширение файла (например, .txt)

                # Ищем свободное имя
                while target_path.exists():
                    new_name = f"{stem}_{counter}{suffix}"
                    target_path = root_dir / new_name
                    counter += 1

                print(f"Конфликт имен: {file_path.name} уже существует. Переименован в {target_path.name}")

            try:
                # Перемещаем файл
                shutil.move(str(file_path), str(target_path))
                print(f"Перемещен: {file_path.relative_to(root_dir)} -> {target_path.name}")
            except Exception as e:
                print(f"Ошибка при перемещении {file_path}: {e}")

    print("Готово!")


def main():
    parser = argparse.ArgumentParser(
        description="Рекурсивно переносит все файлы из поддиректорий в указанную корневую директорию."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Путь к целевой директории"
    )

    args = parser.parse_args()

    dir_path = Path(args.directory)
    move_files_to_root(dir_path)


if __name__ == "__main__":
    main()