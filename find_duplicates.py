import os
import sys
import imagehash
from PIL import Image
import html
import numpy as np
from scipy.spatial import distance
import torch
from transformers import AutoImageProcessor, AutoModel

# --- Настройки ---
THRESHOLD = 10  # Порог схожести хэшей (phash)
VECTOR_THRESHOLD = 0.3  # Порог косинусного расстояния для ViT
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
HTML_REPORT_NAME = "duplicates_report.html"
MODEL_NAME = "google/vit-large-patch16-384"


# -----------------

def load_vit_model():
    """Загружает модель ViT с Hugging Face."""
    print(f"Загрузка модели {MODEL_NAME}...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        print(f"Модель загружена на устройство: {device}")
        return processor, model, device
    except Exception as e:
        print(f"Критическая ошибка загрузки модели: {e}")
        sys.exit(1)


def get_local_files(directory):
    """Получает список файлов только в текущей директории."""
    files = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path) and item.lower().endswith(SUPPORTED_EXTENSIONS):
            size = os.path.getsize(full_path)
            files.append({'name': item, 'path': os.path.abspath(full_path), 'size': size})
    return files


def calculate_hashes(files_data):
    """Быстрое вычисление хэшей для всех файлов."""
    data = []
    print(f"Слой 1: Вычисление хэшей для {len(files_data)} файлов...")

    for i, item in enumerate(files_data):
        try:
            sys.stdout.write(f"\rОбработано: {i + 1}/{len(files_data)}")
            sys.stdout.flush()

            img = Image.open(item['path'])
            h = imagehash.phash(img)

            # Копируем словарь и добавляем хэш
            item['hash'] = h
            data.append(item)

        except Exception as e:
            print(f"\n[!] Ошибка с файлом {item['name']}: {e}")

    print("\nВычисление хэшей завершено.")
    return data


def find_candidate_groups_by_hash(files_with_hashes):
    """
    Первый слой проверки. Находит группы кандидатов только по хэшу.
    Возвращает список групп (списки словарей файлов).
    """
    graph = {item['path']: [] for item in files_with_hashes}
    info_map = {item['path']: item for item in files_with_hashes}
    paths = list(graph.keys())
    n = len(paths)

    print(f"Поиск кандидатов по хэшу (порог {THRESHOLD})...")

    for i in range(n):
        for j in range(i + 1, n):
            p1 = paths[i]
            p2 = paths[j]

            h1 = info_map[p1]['hash']
            h2 = info_map[p2]['hash']

            if (h1 - h2) <= THRESHOLD:
                graph[p1].append(p2)
                graph[p2].append(p1)

    # Собираем компоненты связности
    visited = set()
    groups = []

    for path in paths:
        if path not in visited:
            stack = [path]
            current_group = []

            while stack:
                curr = stack.pop()
                if curr in visited: continue

                visited.add(curr)
                current_group.append(info_map[curr])

                for neighbor in graph[curr]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            if len(current_group) > 1:
                groups.append(current_group)

    return groups


def calculate_vectors_for_candidates(groups, processor, model, device):
    """
    Второй слой: вычисляет векторы ТОЛЬКО для файлов из групп-кандидатов.
    Обновляет словари файлов внутри групп, добавляя поле 'vector'.
    """
    # Собираем уникальные пути из всех групп, чтобы не считать одно фото дважды
    unique_paths = set()
    for group in groups:
        for item in group:
            unique_paths.add(item['path'])

    print(f"\nСлой 2: Вычисление векторов ViT для {len(unique_paths)} кандидатов...")

    # Мапа путь -> вектор для кэширования
    vectors_cache = {}

    count = 0
    total = len(unique_paths)

    for path in unique_paths:
        try:
            count += 1
            sys.stdout.write(f"\rВекторы: {count}/{total}")
            sys.stdout.flush()

            img = Image.open(path).convert('RGB')

            with torch.no_grad():
                inputs = processor(images=img, return_tensors="pt").to(device)
                outputs = model(**inputs)
                embedding = outputs.pooler_output[0].cpu().numpy()
                embedding = embedding / np.linalg.norm(embedding)

            vectors_cache[path] = embedding

        except Exception as e:
            print(f"\n[!] Ошибка векторизации {path}: {e}")
            vectors_cache[path] = None

    print("\nВекторизация завершена.")

    # Распределяем векторы обратно по группам
    for group in groups:
        for item in group:
            item['vector'] = vectors_cache.get(item['path'])

    return groups


def refine_groups_with_vectors(groups):
    """
    Точная проверка найденных групп векторами.
    Если внутри группы векторное расстояние велико, группа распадается.
    """
    print(f"Уточнение групп векторами (порог {VECTOR_THRESHOLD})...")

    final_groups = []

    for group in groups:
        # Строим подграф для текущей группы с учетом векторов
        graph = {item['path']: [] for item in group}

        # Сравниваем все пары внутри группы
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                item1 = group[i]
                item2 = group[j]

                # Если у кого-то нет вектора - считаем связь валидной (даем пользу сомнения)
                # или разрываем, если нужна 100% точность. Здесь даем пользу сомнения.
                if item1.get('vector') is None or item2.get('vector') is None:
                    graph[item1['path']].append(item2['path'])
                    graph[item2['path']].append(item1['path'])
                    continue

                dist_vector = distance.cosine(item1['vector'], item2['vector'])

                if dist_vector <= VECTOR_THRESHOLD:
                    graph[item1['path']].append(item2['path'])
                    graph[item2['path']].append(item1['path'])

        # Ищем компоненты связности внутри этой подгруппы
        visited = set()
        for item in group:
            path = item['path']
            if path not in visited:
                stack = [path]
                current_final_group = []

                while stack:
                    curr = stack.pop()
                    if curr in visited: continue

                    # Находим исходный объект по пути
                    curr_item = next((x for x in group if x['path'] == curr), None)
                    if curr_item:
                        visited.add(curr)
                        current_final_group.append(curr_item)

                        for neighbor in graph[curr]:
                            if neighbor not in visited:
                                stack.append(neighbor)

                if len(current_final_group) > 1:
                    final_groups.append(current_final_group)

    return final_groups


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / 1024 ** 2:.1f} MB"


def generate_html_report(groups, output_dir):
    # ... (без изменений, код генерации HTML тот же что и раньше) ...
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Отчет о дубликатах (ViT Optimized)</title>
        <style>
            body { font-family: sans-serif; background: #f4f4f4; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            .group { border: 2px solid #ddd; padding: 15px; margin-bottom: 25px; border-radius: 8px; background: #fff; }
            .group-header { font-weight: bold; margin-bottom: 10px; color: #d9534f; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            .images-row { display: flex; flex-wrap: wrap; gap: 15px; align-items: flex-start; }
            .image-block { text-align: center; border: 1px solid #eee; padding: 5px; background: #fafafa; border-radius: 4px; max-width: 220px; }
            img { max-width: 200px; max-height: 200px; display: block; margin-bottom: 5px; object-fit: cover; cursor: pointer; }
            img:hover { opacity: 0.8; }
            .image-block.first { background: #e8f5e9; border-color: #c8e6c9; }
            .controls { background: #333; color: white; padding: 20px; position: sticky; bottom: 0; text-align: center; margin-top: 20px; border-radius: 8px; }
            button { padding: 12px 24px; font-size: 16px; cursor: pointer; background: #28a745; color: white; border: none; border-radius: 4px; font-weight: bold; }
            button:hover { background: #218838; }
            textarea { width: 100%; height: 150px; margin-top: 15px; font-family: monospace; display: none; background: #fff; color: #000; border: 1px solid #ccc; }
            label { display: block; font-size: 11px; color: #555; word-break: break-all; margin-top: 5px; }
            .size-label { color: #007bff; font-weight: bold; font-size: 12px; }
        </style>
    </head>
    <body>
    <div class="container">
        <h1>Отчет о дубликатах</h1>
        <p>Найдено групп: """ + str(len(groups)) + """</p>
        <p><small>Модель: """ + MODEL_NAME + """. Этапы: Hash (быстрый) -> ViT (точный для кандидатов).</small></p>
        <div class="controls">
            <button onclick="generateBashCommands()">Удалить отмеченное</button>
            <textarea id="bashOutput"></textarea>
        </div>
        <div class="groups-list">
    """
    for i, group in enumerate(groups):
        group.sort(key=lambda x: x['size'], reverse=True)
        html_content += f'<div class="group"><div class="group-header">Группа #{i + 1} ({len(group)} файлов)</div><div class="images-row">'
        for idx, file_info in enumerate(group):
            path = file_info['path'];
            name = file_info['name'];
            size = file_info['size']
            file_uri = ('file:///' + path.replace(os.sep, '/')) if os.name == 'nt' else ('file://' + path)
            safe_val = html.escape(path);
            safe_uri = html.escape(file_uri);
            safe_name = html.escape(name)
            block_class = "image-block first" if idx == 0 else "image-block"
            html_content += f"""
                <div class="{block_class}">
                    <img src="{safe_uri}" alt="Image" onerror="this.src=''; this.alt='Ошибка';">
                    <div class="size-label">{format_size(size)}</div>
                    <label><input type="checkbox" class="del-checkbox" value="{safe_val}"> {safe_name}</label>
                </div>"""
        html_content += "</div></div>"

    html_content += """
        </div>
    </div>
    <script>
        function generateBashCommands() {
            const checkboxes = document.querySelectorAll('.del-checkbox:checked');
            const outputArea = document.getElementById('bashOutput');
            if (checkboxes.length === 0) { alert('Ничего не выбрано!'); outputArea.style.display = 'none'; return; }
            let commands = "#!/bin/bash\\n";
            checkboxes.forEach(cb => {
                let safePath = cb.value.replace(/'/g, "'\\\\''"); 
                commands += "rm '" + safePath + "'\\n";
            });
            outputArea.value = commands; outputArea.style.display = 'block'; outputArea.select();
        }
    </script>
    </body>
    </html>
    """
    report_path = os.path.join(output_dir, HTML_REPORT_NAME)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return report_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        target_dir = os.getcwd()
        print(f"Путь не указан, используется текущая директория: {target_dir}")
    else:
        target_dir = sys.argv[1]

    if not os.path.isdir(target_dir):
        print(f"Ошибка: Директория '{target_dir}' не существует.")
        sys.exit(1)

    # 1. Получаем список файлов
    files = get_local_files(target_dir)
    if not files:
        print("Изображения не найдены.")
        sys.exit(0)

    # 2. Слой 1: Считаем хэши
    hashed_files = calculate_hashes(files)

    # 3. Слой 1: Ищем кандидатов (без загрузки модели!)
    candidate_groups = find_candidate_groups_by_hash(hashed_files)

    if not candidate_groups:
        print("Дубликаты не найдены на первом этапе.")
        sys.exit(0)

    print(f"\nНайдено групп кандидатов по хэшу: {len(candidate_groups)}")

    # 4. Слой 2: Загружаем тяжелую модель ТОЛЬКО если есть кандидаты
    processor, model, device = load_vit_model()

    # 5. Слой 2: Считаем векторы только для кандидатов
    candidate_groups = calculate_vectors_for_candidates(candidate_groups, processor, model, device)

    # 6. Слой 2: Уточняем группы векторами
    final_groups = refine_groups_with_vectors(candidate_groups)

    if not final_groups:
        print("После точечной проверки векторами дубликаты не найдены.")
        sys.exit(0)

    report_path = generate_html_report(final_groups, target_dir)

    print(f"\nГотово! Найдено подтвержденных групп дубликатов: {len(final_groups)}")
    print(f"HTML отчет сохранен: {report_path}")