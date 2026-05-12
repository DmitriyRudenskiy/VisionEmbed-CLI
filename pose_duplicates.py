#!/usr/bin/env python3
"""
Поиск дубликатов изображений по позе человека с помощью DWPose.
Нормализация по длине торса, взвешенный RMSE, ограничение на отклонение одного сустава.
"""
import os
import sys
import json
import math
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
import html

from dwpose import DwposeDetector

# ------------------- Глобальные значения по умолчанию -------------------
DEFAULT_POSE_THRESHOLD = 0.07
DEFAULT_MAX_JOINT_DIST = 0.12
DEFAULT_MIN_COMMON_JOINTS = 10
DWPOSE_RES = 1024
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
DEFAULT_HTML_REPORT = "duplicates_report.html"
POSE_CACHE_FILE = "pose_cache.json"

# Веса и названия суставов (COCO body, 18 точек)
JOINT_NAMES = [
    "Nose", "Neck", "R.Shoulder", "R.Elbow", "R.Wrist",
    "L.Shoulder", "L.Elbow", "L.Wrist", "R.Hip", "R.Knee",
    "R.Ankle", "L.Hip", "L.Knee", "L.Ankle", "R.Eye", "L.Eye", "R.Ear", "L.Ear"
]
JOINT_WEIGHTS = [0.8, 1.2, 1.0, 0.8, 0.5, 1.0, 0.8, 0.5,
                 1.2, 0.9, 0.6, 1.2, 0.9, 0.6, 0.4, 0.4, 0.4, 0.4]

log = logging.getLogger("pose_dup")

# ------------------- Вспомогательные функции -------------------
def setup_logging(debug: bool):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    log.handlers.clear()
    log.addHandler(handler)

def pad_to_square(img: Image.Image, target_size: int = DWPOSE_RES) -> Image.Image:
    img = img.convert("RGB")
    old_w, old_h = img.size
    ratio = min(target_size / old_w, target_size / old_h)
    new_w = int(old_w * ratio)
    new_h = int(old_h * ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    new_img.paste(img, (pad_x, pad_y))
    return new_img

def load_dwpose_model() -> DwposeDetector:
    log.info("Загрузка модели DWPose...")
    try:
        model = DwposeDetector.from_pretrained_default()
        log.info("Модель загружена.")
        return model
    except Exception as e:
        log.critical("Не удалось загрузить модель DWPose: %s", e)
        sys.exit(1)

# ------------------- Извлечение позы -------------------
def extract_normalized_pose(image_path: str, model: DwposeDetector) -> list[dict] | None:
    try:
        log.debug("Обработка: %s", os.path.basename(image_path))
        img = Image.open(image_path)
        img_squared = pad_to_square(img)

        _, j, _ = model(
            img_squared,
            include_hand=False,
            include_face=False,
            include_body=True,
            image_and_json=True,
            detect_resolution=DWPOSE_RES
        )

        if not j or not j.get("people"):
            log.debug("Поза не обнаружена.")
            return None

        kp = j["people"][0].get("pose_keypoints_2d", [])
        points = []
        for i in range(0, len(kp), 3):
            points.append({'x': kp[i], 'y': kp[i+1], 'conf': kp[i+2]})

        neck = points[1]
        rhip = points[8]
        lhip = points[11]

        if neck['conf'] == 0 or (rhip['conf'] == 0 and lhip['conf'] == 0):
            log.debug("Торс не распознан.")
            return None

        hips = []
        if rhip['conf'] > 0:
            hips.append((rhip['x'], rhip['y']))
        if lhip['conf'] > 0:
            hips.append((lhip['x'], lhip['y']))
        mid_hip_x = sum(h[0] for h in hips) / len(hips)
        mid_hip_y = sum(h[1] for h in hips) / len(hips)

        center_x = (neck['x'] + mid_hip_x) / 2
        center_y = (neck['y'] + mid_hip_y) / 2

        torso_len = math.hypot(neck['x'] - mid_hip_x, neck['y'] - mid_hip_y)
        if torso_len == 0:
            return None

        for p in points:
            p['x'] = (p['x'] - center_x) / torso_len
            p['y'] = (p['y'] - center_y) / torso_len

        if log.isEnabledFor(logging.DEBUG):
            debug_joints = {1: "Neck", 3: "R.Elbow", 6: "L.Elbow", 9: "R.Knee", 12: "L.Knee"}
            parts = []
            for idx, name in debug_joints.items():
                if points[idx]['conf'] > 0:
                    parts.append(f"{name}:({points[idx]['x']:.2f},{points[idx]['y']:.2f})")
                else:
                    parts.append(f"{name}:NONE")
            log.debug("Норм. координаты: %s", " | ".join(parts))

        return points

    except Exception as e:
        log.error("Ошибка в %s: %s", image_path, e)
        return None

# ------------------- Расстояние между позами -------------------
def calculate_pose_distance(
    pose1: list[dict],
    pose2: list[dict],
    min_common_joints: int,
    max_joint_dist: float,
    name1: str = "",
    name2: str = "",
    debug_threshold: float = None
) -> float:
    """
    Взвешенный RMSE и проверка предельного смещения сустава.
    Возвращает RMSE или float('inf'), если позы слишком разные.
    """
    max_dist = 0.0
    max_joint_name = ""
    weighted_sq_sum = 0.0
    weight_sum = 0.0
    valid = 0

    min_len = min(len(pose1), len(pose2))
    for i in range(min_len):
        p1, p2 = pose1[i], pose2[i]
        if p1['conf'] <= 0.3 or p2['conf'] <= 0.3:
            continue

        dx = p1['x'] - p2['x']
        dy = p1['y'] - p2['y']
        dist = math.hypot(dx, dy)

        w = JOINT_WEIGHTS[i] if i < len(JOINT_WEIGHTS) else 0.8
        weighted_sq_sum += w * dist * dist
        weight_sum += w
        valid += 1

        if dist > max_dist:
            max_dist = dist
            max_joint_name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"J{i}"

    if valid < min_common_joints:
        log.debug("%s vs %s: мало общих точек (%d)", name1, name2, valid)
        return float('inf')

    if max_dist > max_joint_dist:
        log.debug("%s vs %s: сустав %s смещён на %.3f (порог %.3f)",
                  name1, name2, max_joint_name, max_dist, max_joint_dist)
        return float('inf')

    rmse = math.sqrt(weighted_sq_sum / weight_sum) if weight_sum > 0 else float('inf')

    if debug_threshold is not None and log.isEnabledFor(logging.DEBUG) and rmse < debug_threshold * 3:
        status = "ДУБЛИКАТ" if rmse <= debug_threshold else "разные"
        log.debug("%s vs %s: RMSE=%.4f, макс=%.3f (%s) [%s]",
                  name1, name2, rmse, max_dist, max_joint_name, status)
    return rmse

# ------------------- Работа с файлами -------------------
def get_image_files(directory: str) -> list[dict]:
    files = []
    dir_path = Path(directory).resolve()
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append({'name': p.name, 'path': str(p), 'size': p.stat().st_size})
    return files

# ------------------- Кеширование поз -------------------
def load_pose_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            log.warning("Не удалось прочитать кеш поз, будет создан новый.")
    return {}

def save_pose_cache(cache: dict, cache_path: str):
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
        log.info("Кеш поз сохранён: %s", cache_path)
    except Exception as e:
        log.warning("Ошибка сохранения кеша: %s", e)

# ------------------- Поиск дубликатов -------------------
def find_duplicates(
    files_data: list[dict],
    model: DwposeDetector,
    pose_threshold: float,
    max_joint_dist: float,
    min_common_joints: int,
    num_workers: int = 1,
    cache_file: str | None = None
) -> list[list[dict]]:
    # Загрузка кеша
    cache = {}
    if cache_file:
        cache = load_pose_cache(cache_file)

    to_process = []
    for item in files_data:
        path = item['path']
        if path in cache:
            item['pose_data'] = cache[path]
        else:
            to_process.append(item)

    if to_process:
        log.info("Извлечение поз для %d изображений (потоков: %d)...", len(to_process), num_workers)
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_item = {executor.submit(extract_normalized_pose, item['path'], model): item for item in to_process}
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        item['pose_data'] = future.result()
                    except Exception as e:
                        log.error("Ошибка потока для %s: %s", item['path'], e)
                        item['pose_data'] = None
        else:
            for item in to_process:
                item['pose_data'] = extract_normalized_pose(item['path'], model)

        if cache_file:
            for item in to_process:
                cache[item['path']] = item['pose_data']
            save_pose_cache(cache, cache_file)
    else:
        log.info("Все позы загружены из кеша.")

    log.info("Попарное сравнение (порог RMSE=%.3f)...", pose_threshold)
    paths = [item['path'] for item in files_data]
    info_by_path = {item['path']: item for item in files_data}
    graph = {p: [] for p in paths}

    n = len(paths)
    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = paths[i], paths[j]
            pose1 = info_by_path[p1].get('pose_data')
            pose2 = info_by_path[p2].get('pose_data')
            if pose1 is None or pose2 is None:
                continue

            dist = calculate_pose_distance(
                pose1, pose2,
                min_common_joints=min_common_joints,
                max_joint_dist=max_joint_dist,
                name1=os.path.basename(p1),
                name2=os.path.basename(p2),
                debug_threshold=pose_threshold
            )
            if dist <= pose_threshold:
                graph[p1].append(p2)
                graph[p2].append(p1)

    visited = set()
    groups = []
    for p in paths:
        if p not in visited:
            stack = [p]
            group = []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                group.append(info_by_path[cur])
                for nb in graph[cur]:
                    if nb not in visited:
                        stack.append(nb)
            if len(group) > 1:
                groups.append(group)

    log.info("Найдено %d групп дубликатов.", len(groups))
    return groups

# ------------------- Генерация отчёта -------------------
def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    else:
        return f"{size_bytes/1024**2:.1f} MB"

def generate_html_report(groups: list[list[dict]], output_dir: str, report_name: str):
    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="ru">',
        "<head>",
        '<meta charset="UTF-8">',
        "<title>Отчёт о дубликатах (DWPose)</title>",
        "<style>",
        "body { font-family: sans-serif; background: #f4f4f4; padding: 20px; }",
        ".container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }",
        "h1 { color: #333; }",
        ".group { border: 2px solid #ddd; padding: 15px; margin-bottom: 25px; border-radius: 8px; background: #fff; }",
        ".group-header { font-weight: bold; margin-bottom: 10px; color: #d9534f; border-bottom: 1px solid #eee; padding-bottom: 5px; }",
        ".images-row { display: flex; flex-wrap: wrap; gap: 15px; align-items: flex-start; }",
        ".image-block { text-align: center; border: 1px solid #eee; padding: 5px; background: #fafafa; border-radius: 4px; max-width: 250px; }",
        "img { max-width: 200px; max-height: 200px; display: block; margin-bottom: 5px; object-fit: cover; }",
        "label { font-size: 12px; display: block; word-break: break-all; }",
        ".size-label { color: #007bff; font-weight: bold; }",
        "</style>",
        "</head>",
        "<body>",
        '<div class="container">',
        "<h1>Отчёт о дубликатах (DWPose RMSE)</h1>",
        f"<p>Найдено групп: {len(groups)}</p>",
        "<p><small>Метод: нормализация по торсу + взвешенный RMSE + ограничение на отклонение сустава.</small></p>",
        '<div class="groups-list">'
    ]

    for i, group in enumerate(groups):
        group.sort(key=lambda x: x['size'], reverse=True)
        html_parts.append(f'<div class="group"><div class="group-header">Группа #{i+1} ({len(group)} файлов)</div><div class="images-row">')
        for file_info in group:
            path = file_info['path']
            name = file_info['name']
            size = file_info['size']
            safe_name = html.escape(name)
            safe_path = html.escape(path)
            if os.name == 'nt':
                file_uri = 'file:///' + path.replace(os.sep, '/')
            else:
                file_uri = 'file://' + path
            safe_uri = html.escape(file_uri)
            html_parts.append(
                f'<div class="image-block">'
                f'<img src="{safe_uri}" alt="?" onerror="this.style.display=\'none\'">'
                f'<div class="size-label">{format_size(size)}</div>'
                f'<label title="{safe_path}">{safe_name}</label>'
                f'</div>'
            )
        html_parts.append('</div></div>')

    html_parts.append('</div></div></body></html>')

    report_path = os.path.join(output_dir, report_name)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    log.info("HTML-отчёт сохранён: %s", report_path)
    return report_path

# ------------------- Главная функция -------------------
def main():
    parser = argparse.ArgumentParser(description="Поиск дубликатов изображений по позе (DWPose).")
    parser.add_argument("directory", nargs="?", default=os.getcwd(),
                        help="Директория с изображениями (по умолчанию текущая).")
    parser.add_argument("--pose-threshold", type=float, default=DEFAULT_POSE_THRESHOLD,
                        help="Порог RMSE для дубликата (по умолчанию %(default).3f).")
    parser.add_argument("--max-joint-dist", type=float, default=DEFAULT_MAX_JOINT_DIST,
                        help="Макс. отклонение одного сустава (по умолчанию %(default).3f).")
    parser.add_argument("--min-common-joints", type=int, default=DEFAULT_MIN_COMMON_JOINTS,
                        help="Мин. число общих точек (по умолчанию %(default)d).")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Количество потоков для извлечения поз.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Не использовать кеш поз.")
    parser.add_argument("--debug", action="store_true",
                        help="Подробный вывод.")
    parser.add_argument("--output", "-o", default=DEFAULT_HTML_REPORT,
                        help="Имя HTML-отчёта (по умолчанию duplicates_report.html).")

    args = parser.parse_args()
    setup_logging(args.debug)

    if not os.path.isdir(args.directory):
        log.error("Директория '%s' не найдена.", args.directory)
        sys.exit(1)

    files = get_image_files(args.directory)
    if not files:
        log.warning("Поддерживаемые изображения не найдены.")
        sys.exit(0)

    model = load_dwpose_model()

    cache_path = None
    if not args.no_cache:
        cache_path = os.path.join(args.directory, POSE_CACHE_FILE)

    groups = find_duplicates(
        files_data=files,
        model=model,
        pose_threshold=args.pose_threshold,
        max_joint_dist=args.max_joint_dist,
        min_common_joints=args.min_common_joints,
        num_workers=args.workers,
        cache_file=cache_path
    )

    if not groups:
        log.info("Дубликаты по позе не найдены.")
        sys.exit(0)

    report_path = generate_html_report(groups, args.directory, args.output)
    print(f"Готово. Отчёт: {report_path}")

if __name__ == "__main__":
    main()