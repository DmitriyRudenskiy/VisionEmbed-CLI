import argparse
import os
from PIL import Image, ImageOps, UnidentifiedImageError

# Поддерживаемые форматы изображений
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif')


def process_image(input_path, output_path, bg_color):
    """Обрабатывает одно изображение: вписывает в квадрат."""
    img = Image.open(input_path)

    # Определяем цвет фона
    if bg_color.lower() == 'transparent':
        img = img.convert('RGBA')
        color = (0, 0, 0, 0)
    else:
        try:
            color = tuple(map(int, bg_color.split(',')))
            if len(color) != 3:
                raise ValueError
        except ValueError:
            raise ValueError("Цвет должен быть в формате R,G,B или 'transparent'")

    # Находим большую сторону
    width, height = img.size
    max_side = max(width, height)

    # Вписываем в квадрат
    square_img = ImageOps.pad(img, (max_side, max_side), color=color, centering=(0.5, 0.5))

    # Конвертация RGBA -> RGB для форматов, не поддерживающих прозрачность (JPG)
    if output_path.lower().endswith(('.jpg', '.jpeg')) and square_img.mode == 'RGBA':
        # Создаем белый фон и накладываем на него изображение
        background = Image.new("RGB", square_img.size, (255, 255, 255))
        background.paste(square_img, mask=square_img.split()[3])  # 3 - канал прозрачности
        square_img = background

    # Сохраняем
    square_img.save(output_path, quality=95)  # quality влияет только на JPG/WEBP
    return max_side


def main():
    parser = argparse.ArgumentParser(description="Вписать все изображения из папки в квадраты по большей стороне.")
    parser.add_argument("input_dir", help="Путь к папке с исходными изображениями")
    parser.add_argument("-o", "--output_dir",
                        help="Путь к папке для сохранения результатов (по умолчанию: <input_dir>_square)", default=None)
    parser.add_argument("-b", "--background", help="Цвет фона (R,G,B) или 'transparent' (по умолчанию: 255,255,255)",
                        default="255,255,255")

    args = parser.parse_args()

    # Проверяем, существует ли входная папка
    if not os.path.isdir(args.input_dir):
        print(f"Ошибка: Папка '{args.input_dir}' не найдена.")
        return

    # Формируем имя выходной папки
    if args.output_dir is None:
        abs_path = os.path.abspath(args.input_dir)
        parent_dir = os.path.dirname(abs_path)
        folder_name = os.path.basename(abs_path)

        if not folder_name:
            folder_name = "output"

        new_folder_name = f"{folder_name}_square"
        output_dir = os.path.join(parent_dir, new_folder_name)
    else:
        output_dir = args.output_dir

    # Создаем выходную папку, если её нет
    os.makedirs(output_dir, exist_ok=True)

    # Считаем статистику
    processed = 0
    skipped = 0

    print(f"Начало обработки папки: {os.path.abspath(args.input_dir)}")
    print(f"Результаты будут сохранены в: {os.path.abspath(output_dir)}\n")

    # Перебираем файлы во входной папке
    for filename in os.listdir(args.input_dir):
        # Игнорируем папки и файлы неподходящих форматов
        if not filename.lower().endswith(VALID_EXTENSIONS):
            continue

        input_path = os.path.join(args.input_dir, filename)

        # Это точно файл, а не подпапка с расширением
        if not os.path.isfile(input_path):
            continue

        output_path = os.path.join(output_dir, filename)

        try:
            max_side = process_image(input_path, output_path, args.background)
            # ИЗМЕНЕНИЕ ЗДЕСЬ: выводим абсолютный путь нового файла
            print(f"[OK] {os.path.abspath(output_path)} -> {max_side}x{max_side}")
            processed += 1
        except UnidentifiedImageError:
            print(f"[ПРОПУСК] {filename} - файл не является изображением или поврежден")
            skipped += 1
        except Exception as e:
            print(f"[ОШИБКА] {filename} - {e}")
            skipped += 1

    print(f"\nГотово! Успешно обработано: {processed}, пропущено/ошибок: {skipped}")


if __name__ == "__main__":
    main()