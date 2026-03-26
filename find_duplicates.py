import os
import sys
import imagehash
from PIL import Image
import html

# --- Настройки ---
THRESHOLD = 10  # Порог схожести
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
HTML_REPORT_NAME = "duplicates_report.html"


# -----------------

def get_local_files(directory):
    """Получает список файлов только в текущей директории."""
    files = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path) and item.lower().endswith(SUPPORTED_EXTENSIONS):
            files.append((item, os.path.abspath(full_path)))
    return files


def calculate_hashes(files_data):
    """Вычисляет хэши для списка файлов."""
    hashes = []
    print(f"Обработка {len(files_data)} файлов...")

    for i, (filename, full_path) in enumerate(files_data):
        try:
            sys.stdout.write(f"\rОбработано: {i + 1}/{len(files_data)}")
            sys.stdout.flush()

            img = Image.open(full_path)
            h = imagehash.phash(img)
            hashes.append({'name': filename, 'path': full_path, 'hash': h})
        except Exception as e:
            print(f"\n[!] Ошибка с файлом {filename}: {e}")

    print("\nВычисление хэшей завершено.")
    return hashes


def find_duplicate_groups(files_with_hashes):
    """
    Находит группы дубликатов (connected components).
    Если A похожа на B, а B на C, они попадут в одну группу [A, B, C].
    """
    # Граф: путь -> список путей соседей
    graph = {item['path']: [] for item in files_with_hashes}

    # Мапа путь -> данные файла (для быстрого доступа)
    info_map = {item['path']: item for item in files_with_hashes}

    paths = list(graph.keys())
    n = len(paths)

    print("Поиск связей между изображениями...")

    # 1. Строим граф связей
    for i in range(n):
        for j in range(i + 1, n):
            p1 = paths[i]
            p2 = paths[j]

            h1 = info_map[p1]['hash']
            h2 = info_map[p2]['hash']

            distance = h1 - h2

            if distance <= THRESHOLD:
                graph[p1].append(p2)
                graph[p2].append(p1)

    # 2. Находим компоненты связности (группы)
    visited = set()
    groups = []

    for path in paths:
        if path not in visited:
            # Обход в глубину (DFS)
            stack = [path]
            current_group = []

            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue

                visited.add(curr)
                current_group.append(info_map[curr])

                for neighbor in graph[curr]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            # Добавляем только если в группе больше 1 файла
            if len(current_group) > 1:
                groups.append(current_group)

    return groups


def generate_html_report(groups, output_dir):
    """Создает HTML файл с группами дубликатов."""

    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Отчет о дубликатах</title>
        <style>
            body { font-family: sans-serif; background: #f4f4f4; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            h1 { color: #333; }

            /* Стили для группы */
            .group { border: 2px solid #ddd; padding: 15px; margin-bottom: 25px; border-radius: 8px; background: #fff; }
            .group-header { font-weight: bold; margin-bottom: 10px; color: #d9534f; border-bottom: 1px solid #eee; padding-bottom: 5px; }

            /* Flexbox для изображений в одну строку с переносом */
            .images-row { display: flex; flex-wrap: wrap; gap: 15px; align-items: flex-start; }

            .image-block { text-align: center; border: 1px solid #eee; padding: 5px; background: #fafafa; border-radius: 4px; max-width: 220px; }
            img { max-width: 200px; max-height: 200px; display: block; margin-bottom: 5px; object-fit: cover; cursor: pointer; }
            img:hover { opacity: 0.8; }

            .controls { background: #333; color: white; padding: 20px; position: sticky; bottom: 0; text-align: center; margin-top: 20px; border-radius: 8px; }
            button { padding: 12px 24px; font-size: 16px; cursor: pointer; background: #28a745; color: white; border: none; border-radius: 4px; font-weight: bold; }
            button:hover { background: #218838; }
            textarea { width: 100%; height: 150px; margin-top: 15px; font-family: monospace; display: none; background: #fff; color: #000; border: 1px solid #ccc; }
            label { display: block; font-size: 11px; color: #555; word-break: break-all; margin-top: 5px; }
        </style>
    </head>
    <body>
    <div class="container">
        <h1>Отчет о дубликатах</h1>
        <p>Найдено групп: """ + str(len(groups)) + """</p>

        <div class="controls">
            <button onclick="generateBashCommands()">Удалить отмеченное (сгенерировать rm)</button>
            <textarea id="bashOutput" placeholder="Команды для удаления..."></textarea>
        </div>

        <div class="groups-list">
    """

    for i, group in enumerate(groups):
        html_content += f"""
        <div class="group">
            <div class="group-header">Группа #{i + 1} ({len(group)} файлов)</div>
            <div class="images-row">
        """

        for file_info in group:
            path = file_info['path']
            name = file_info['name']

            # Формируем корректный file:// URL
            if os.name == 'nt':
                file_uri = 'file:///' + path.replace(os.sep, '/')
            else:
                file_uri = 'file://' + path

            # Экранирование
            safe_val = html.escape(path)
            safe_uri = html.escape(file_uri)
            safe_name = html.escape(name)

            html_content += f"""
                <div class="image-block">
                    <img src="{safe_uri}" alt="Image" onerror="this.src=''; this.alt='Ошибка загрузки';">
                    <label>
                        <input type="checkbox" class="del-checkbox" value="{safe_val}">
                        {safe_name}
                    </label>
                </div>
            """

        html_content += """
            </div>
        </div>
        """

    html_content += """
        </div>
    </div>

    <script>
        function generateBashCommands() {
            const checkboxes = document.querySelectorAll('.del-checkbox:checked');
            const outputArea = document.getElementById('bashOutput');

            if (checkboxes.length === 0) {
                alert('Ничего не выбрано!');
                outputArea.style.display = 'none';
                return;
            }

            // Просто rm, без флагов -iv
            let commands = "#!/bin/bash\\n";

            checkboxes.forEach(cb => {
                // Экранируем одинарные кавычки для bash
                let safePath = cb.value.replace(/'/g, "'\\\\''"); 
                commands += "rm '" + safePath + "'\\n";
            });

            outputArea.value = commands;
            outputArea.style.display = 'block';
            outputArea.select();
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

    files = get_local_files(target_dir)

    if not files:
        print("Изображения не найдены.")
        sys.exit(0)

    hashed_files = calculate_hashes(files)

    # Используем новую функцию группировки
    groups = find_duplicate_groups(hashed_files)

    if not groups:
        print("Дубликаты не найдены.")
        sys.exit(0)

    report_path = generate_html_report(groups, target_dir)

    print(f"\nГотово! Найдено групп дубликатов: {len(groups)}")
    print(f"HTML отчет сохранен: {report_path}")