#!/usr/bin/env python3
"""
Поиск дубликатов изображений по позе человека с помощью DWPose.
Нормализация по длине торса, взвешенный RMSE, ограничение на отклонение сустава.
"""
import os
import sys
import json
import logging
import argparse
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import html
import base64

try:
    from dwpose import DwposeDetector
except ImportError:
    print("❌ Ошибка: модуль 'dwpose' не найден. Установите его или активируйте правильное окружение.")
    sys.exit(1)

# ------------------- Константы -------------------
DEFAULT_POSE_THRESHOLD = 0.07
DEFAULT_MAX_JOINT_DIST = 0.12
DEFAULT_MIN_COMMON_JOINTS = 10
DWPOSE_RES = 1024
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
DEFAULT_HTML_REPORT = "duplicates_report.html"
POSE_CACHE_FILE = "pose_cache.json"

JOINT_NAMES = np.array([
    "Nose", "Neck", "R.Shoulder", "R.Elbow", "R.Wrist",
    "L.Shoulder", "L.Elbow", "L.Wrist", "R.Hip", "R.Knee",
    "R.Ankle", "L.Hip", "L.Knee", "L.Ankle", "R.Eye", "L.Eye", "R.Ear", "L.Ear"
])
JOINT_WEIGHTS = np.array([
    0.8, 1.2, 1.0, 0.8, 0.5, 1.0, 0.8, 0.5,
    1.2, 0.9, 0.6, 1.2, 0.9, 0.6, 0.4, 0.4, 0.4, 0.4
])

# Base64 placeholder for missing images
PLACEHOLDER_IMG_B64 = base64.b64encode(
    b'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
    b'<rect width="100" height="100" fill="#eee"/>'
    b'<text x="50" y="50" dominant-baseline="middle" text-anchor="middle" fill="#999">No preview</text></svg>'
).decode('utf-8')

log = logging.getLogger("pose_dup")
_model_lock = threading.Lock()


# ------------------- Логирование -------------------
def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


# ------------------- Работа с изображениями -------------------
def prepare_image(img: Image.Image, max_size: int = DWPOSE_RES) -> Image.Image:
    """Пропорциональное уменьшение изображения без черных полей."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def load_dwpose_model() -> DwposeDetector:
    log.info("Загрузка модели DWPose...")
    try:
        model = DwposeDetector.from_pretrained_default()
        log.info("✅ Модель успешно загружена.")
        return model
    except Exception as e:
        raise RuntimeError(f"Не удалось загрузить DWPose: {e}") from e


# ------------------- Извлечение позы -------------------
def extract_normalized_pose(image_path: Path, model: DwposeDetector) -> Optional[np.ndarray]:
    """Возвращает массив (18, 3) [x, y, conf] или None."""
    try:
        log.debug("Обработка: %s", image_path.name)

        # I/O операции вынесены за пределы блокировки модели
        img = Image.open(image_path)
        img_prep = prepare_image(img)

        with _model_lock:  # PyTorch не всегда потокобезопасен
            _, j, _ = model(
                img_prep, include_hand=False, include_face=False, include_body=True,
                image_and_json=True, detect_resolution=DWPOSE_RES
            )

        if not j or not j.get("people"):
            log.debug("Поза не обнаружена в %s", image_path.name)
            return None

        kp = j["people"][0].get("pose_keypoints_2d", [])
        pose = np.array(kp, dtype=np.float32).reshape(-1, 3)  # (18, 3)

        neck = pose[1]
        rhip, lhip = pose[8], pose[11]

        if neck[2] <= 0.3 or (rhip[2] <= 0.3 and lhip[2] <= 0.3):
            log.debug("Торс не распознан в %s", image_path.name)
            return None

        hip_mask = np.array([rhip[2], lhip[2]]) > 0.3
        hips = np.array([rhip[:2], lhip[:2]])[hip_mask]
        mid_hip = hips.mean(axis=0)
        center = (neck[:2] + mid_hip) / 2.0
        torso_len = np.linalg.norm(neck[:2] - mid_hip)

        if torso_len < 1e-4:
            return None

        pose[:, :2] = (pose[:, :2] - center) / torso_len
        return pose

    except Exception as e:
        log.error("Ошибка обработки %s: %s", image_path.name, e)
        return None


# ------------------- Кеширование -------------------
def load_pose_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            return {k: v for k, v in raw.items()}  # Возвращаем как есть (внутри pose, mtime, size)
        except Exception as e:
            log.warning("Не удалось прочитать кеш: %s", e)
    return {}


def save_pose_cache(files_data: list[dict], cache_path: Path) -> None:
    try:
        serializable = {}
        for item in files_data:
            pose_data = item.get("pose_data")
            path_str = str(item["path"])

            if pose_data is not None:
                serializable[path_str] = {
                    "pose": pose_data.tolist() if isinstance(pose_data, np.ndarray) else pose_data,
                    "mtime": item["mtime"],
                    "size": item["size"]
                }

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        log.info("💾 Кеш поз сохранён: %s", cache_path)
    except Exception as e:
        log.warning("Ошибка сохранения кеша: %s", e)


# ------------------- Поиск файлов -------------------
def get_image_files(directory: Path) -> list[dict]:
    files = []
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            stat = p.stat()
            files.append({
                "name": p.name,
                "path": p.resolve(),  # Сразу сохраняем абсолютные пути
                "size": stat.st_size,
                "mtime": stat.st_mtime  # Сохраняем время модификации сразу
            })
    return files


# ------------------- Поиск дубликатов -------------------
def find_duplicates(
        files_data: list[dict], model: DwposeDetector,
        pose_threshold: float, max_joint_dist: float, min_common_joints: int,
        num_workers: int = 1, cache_file: Optional[Path] = None
) -> list[list[dict]]:
    cache = load_pose_cache(cache_file) if cache_file else {}
    to_process = []

    # Проверка кеша
    for item in files_data:
        path_str = str(item["path"])
        if path_str in cache:
            entry = cache[path_str]
            # Проверяем актуальность (сравниваем с тем, что сохранили при обходе)
            if entry.get("mtime") == item["mtime"] and entry.get("size") == item["size"]:
                item["pose_data"] = np.array(entry["pose"], dtype=np.float32) if entry.get("pose") else None
                continue
        to_process.append(item)

    # Извлечение новых поз
    if to_process:
        log.info("Извлечение поз для %d изображений (потоков: %d)...", len(to_process), num_workers)
        if num_workers > 1:
            log.warning(
                "⚠️ Многопоточность с GPU-моделями может вызывать ошибки CUDA. При сбоях используйте --workers 1")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_item = {
                executor.submit(extract_normalized_pose, item["path"], model): item
                for item in to_process
            }
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    item["pose_data"] = future.result()
                except Exception as e:
                    log.error("Поток завершился с ошибкой для %s: %s", item["path"].name, e)
                    item["pose_data"] = None

        if cache_file:
            save_pose_cache(files_data, cache_file)
    else:
        log.info("✅ Все позы загружены из актуального кеша.")

    # Оптимизированное попарное сравнение O(N^2)
    log.info("Попарное сравнение (порог RMSE=%.3f)...", pose_threshold)
    valid_items = [it for it in files_data if it.get("pose_data") is not None]

    # Извлекаем позы в плоский массив для скорости
    poses = np.array([it["pose_data"] for it in valid_items])
    conf_masks = poses[:, :, 2] > 0.3

    graph = defaultdict(set)
    n = len(valid_items)

    for i in range(n):
        p1 = poses[i]
        mask1 = conf_masks[i]
        name1 = valid_items[i]["name"]

        for j in range(i + 1, n):
            p2 = poses[j]
            mask2 = conf_masks[j]

            common_mask = mask1 & mask2
            valid_count = common_mask.sum()

            if valid_count < min_common_joints:
                continue

            diffs = p1[common_mask, :2] - p2[common_mask, :2]
            dists = np.linalg.norm(diffs, axis=1)
            max_dist = dists.max()

            if max_dist > max_joint_dist:
                if log.isEnabledFor(logging.DEBUG):
                    idx = np.argmax(dists)
                    joint_idx = np.where(common_mask)[0][idx]
                    log.debug("%s vs %s: сустав %s смещён на %.3f (порог %.3f)",
                              name1, valid_items[j]["name"], JOINT_NAMES[joint_idx], max_dist, max_joint_dist)
                continue

            weights = JOINT_WEIGHTS[common_mask]
            rmse = np.sqrt(np.sum(weights * dists ** 2) / np.sum(weights))

            if rmse <= pose_threshold:
                path1, path2 = valid_items[i]["path"], valid_items[j]["path"]
                graph[path1].add(path2)
                graph[path2].add(path1)

            if log.isEnabledFor(logging.DEBUG) and rmse < pose_threshold * 3:
                status = "ДУБЛИКАТ" if rmse <= pose_threshold else "разные"
                log.debug("%s vs %s: RMSE=%.4f, макс=%.3f [%s]", name1, valid_items[j]["name"], rmse, max_dist, status)

    # Объединение в группы (DFS)
    visited = set()
    groups = []
    info_by_path = {it["path"]: it for it in valid_items}

    for p in info_by_path:
        if p not in visited:
            stack = [p]
            group = []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                group.append(info_by_path[cur])
                stack.extend(nb for nb in graph[cur] if nb not in visited)
            if len(group) > 1:
                groups.append(group)

    log.info("🎯 Найдено %d групп дубликатов.", len(groups))
    return groups


# ------------------- HTML-отчёт -------------------
def format_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def generate_html_report(groups: list[list[dict]], output_dir: Path, report_name: str) -> Path:
    html_parts = [
        "<!DOCTYPE html><html lang='ru'><head><meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<title>Отчёт о дубликатах (DWPose)</title>",
        "<style>",
        ":root { --bg: #f4f6f9; --card: #fff; --text: #333; --accent: #007bff; --border: #e0e0e0; }",
        "@media (prefers-color-scheme: dark) { :root { --bg: #121212; --card: #1e1e1e; --text: #e0e0e0; --accent: #4da3ff; --border: #333; } }",
        "body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; }",
        ".container { max-width: 1200px; margin: 0 auto; }",
        "h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }",
        ".meta { color: #888; font-size: 0.9rem; margin-bottom: 20px; }",
        ".group { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 20px; }",
        ".group-header { font-weight: 600; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }",
        ".card { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 8px; text-align: center; }",
        "img { max-width: 100%; height: 180px; object-fit: cover; border-radius: 6px; margin-bottom: 6px; }",
        ".info { font-size: 0.85rem; word-break: break-all; }",
        ".size { color: var(--accent); font-weight: 500; }",
        "button { background: var(--accent); color: #fff; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.8rem; margin-top: 4px; }",
        "button:hover { opacity: 0.9; }",
        "</style></head><body><div class='container'>",
        "<h1>🔍 Отчёт о дубликатах по позе</h1>",
        f"<div class='meta'>Найдено групп: <b>{len(groups)}</b> | Метод: нормализация торса + взвешенный RMSE</div>"
    ]

    for i, group in enumerate(groups, 1):
        group.sort(key=lambda x: x["size"], reverse=True)
        html_parts.append(
            f"<div class='group'><div class='group-header'>Группа #{i} ({len(group)} файлов)</div><div class='grid'>")
        for f in group:
            path_str = str(f["path"])
            uri = f"file:///{path_str.replace(os.sep, '/')}" if os.name == 'nt' else f"file://{path_str}"

            # Безопасное экранирование
            safe_uri = html.escape(uri)
            safe_name = html.escape(f['name'])
            safe_path_btn = html.escape(path_str)

            html_parts.append(
                f"<div class='card'>"
                f"<img src='{safe_uri}' alt='preview' onerror=\"this.onerror=null;this.src='data:image/svg+xml;base64,{PLACEHOLDER_IMG_B64}'\">"
                f"<div class='info'><span class='size'>{format_size(f['size'])}</span><br>{safe_name}</div>"
                f"<button onclick=\"navigator.clipboard.writeText('{safe_path_btn}').then(()=>this.textContent='Скопировано!')\">📋 Путь</button>"
                f"</div>"
            )
        html_parts.append("</div></div>")

    html_parts.append("</div></body></html>")

    report_path = output_dir / report_name
    report_path.write_text("\n".join(html_parts), encoding="utf-8")
    log.info("📄 HTML-отчёт сохранён: %s", report_path)
    return report_path


# ------------------- CLI & Main -------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Поиск дубликатов изображений по позе (DWPose).")
    parser.add_argument("directory", nargs="?", default=".", help="Директория с изображениями.")
    parser.add_argument("--pose-threshold", type=float, default=DEFAULT_POSE_THRESHOLD,
                        help="Порог RMSE для дубликата.")
    parser.add_argument("--max-joint-dist", type=float, default=DEFAULT_MAX_JOINT_DIST,
                        help="Макс. отклонение одного сустава.")
    parser.add_argument("--min-common-joints", type=int, default=DEFAULT_MIN_COMMON_JOINTS,
                        help="Мин. число общих точек.")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Потоки для извлечения поз (рекомендуется 1 для GPU).")
    parser.add_argument("--no-cache", action="store_true", help="Отключить кеширование поз.")
    parser.add_argument("--debug", action="store_true", help="Подробный вывод.")
    parser.add_argument("--output", "-o", default=DEFAULT_HTML_REPORT, help="Имя HTML-отчёта.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)

    target_dir = Path(args.directory).resolve()
    if not target_dir.is_dir():
        log.error("❌ Директория '%s' не найдена.", target_dir)
        sys.exit(1)

    files = get_image_files(target_dir)
    if not files:
        log.warning("⚠️ Поддерживаемые изображения не найдены.")
        sys.exit(0)

    try:
        model = load_dwpose_model()
    except RuntimeError as e:
        log.critical(str(e))
        sys.exit(1)

    cache_path = None if args.no_cache else target_dir / POSE_CACHE_FILE

    try:
        groups = find_duplicates(
            files_data=files, model=model,
            pose_threshold=args.pose_threshold, max_joint_dist=args.max_joint_dist,
            min_common_joints=args.min_common_joints, num_workers=args.workers,
            cache_file=cache_path
        )
    except KeyboardInterrupt:
        log.warning("⛔ Прервано пользователем.")
        sys.exit(130)
    except Exception as e:
        log.critical("💥 Критическая ошибка: %s", e, exc_info=args.debug)
        sys.exit(1)

    if not groups:
        log.info("✅ Дубликаты по позе не найдены.")
        sys.exit(0)

    report_path = generate_html_report(groups, target_dir, args.output)
    print(f"\n✅ Готово. Отчёт: {report_path}")


if __name__ == "__main__":
    main()