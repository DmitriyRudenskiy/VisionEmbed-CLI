import argparse
import os
import json
from PIL import Image


def get_palette_json(img, color_count=20):
    # Обработка прозрачности
    if img.mode in ('RGBA', 'LA', 'P'):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        if img.mode in ('RGBA', 'LA'):
            background.paste(img, mask=img.split()[-1])
            img = background
        else:
            img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    quantized = img.quantize(colors=color_count, method=2, dither=0)
    color_counts = quantized.getcolors()
    palette_data = quantized.getpalette()
    total_pixels = img.size[0] * img.size[1]

    results = []
    for count, idx in color_counts:
        r = palette_data[idx * 3]
        g = palette_data[idx * 3 + 1]
        b = palette_data[idx * 3 + 2]
        percent = round((count / total_pixels) * 100, 2)

        results.append({
            "rgb": [r, g, b],
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "percent": percent,
            "_count": count
        })

    results.sort(key=lambda x: x["_count"], reverse=True)
    for item in results:
        del item["_count"]

    return results


def process_images(target_path):
    output_data = []
    files_to_process = []

    if os.path.isfile(target_path):
        files_to_process.append(target_path)
    elif os.path.isdir(target_path):
        supported_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff')
        files_to_process = [
            os.path.join(target_path, f) for f in os.listdir(target_path)
            if f.lower().endswith(supported_exts)
        ]
    else:
        return {"error": "Path does not exist"}

    for file_path in files_to_process:
        try:
            img = Image.open(file_path)
            colors = get_palette_json(img, color_count=20)

            # Возвращаем абсолютный путь
            abs_path = os.path.abspath(file_path)

            output_data.append({
                "filename": abs_path,
                "colors": colors
            })
        except Exception as e:
            output_data.append({
                "filename": file_path,
                "error": str(e)
            })

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Extract colors.")
    parser.add_argument("path", help="Image or Directory")
    args = parser.parse_args()
    result = process_images(args.path)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()