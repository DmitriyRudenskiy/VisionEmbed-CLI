#!/usr/bin/env python3
"""
Поиск дубликатов изображений по позе человека с помощью DWPose (ООП-версия).
Нормализация по длине торса, взвешенный RMSE, ограничение на отклонение сустава.
Результат сохраняется в JSON файл.
"""
import os
import sys
import json
import logging
import argparse
import threading
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
from PIL import Image, UnidentifiedImageError, ImageOps

try:
    from dwpose import DwposeDetector
except ImportError:
    print(
        "❌ Ошибка: модуль 'dwpose' не найден. "
        "Установите его или активируйте правильное окружение."
    )
    sys.exit(1)

# ------------------- Константы -------------------
DEFAULT_POSE_THRESHOLD = 0.07
DEFAULT_MAX_JOINT_DIST = 0.12
DEFAULT_MIN_COMMON_JOINTS = 10
DWPOSE_RES = 1024
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
DEFAULT_JSON_REPORT = "duplicates.json"
POSE_CACHE_FILE = "pose_cache.json"
CACHE_VERSION = 1
MIN_CONFIDENCE = 0.3
MIN_TORSO_CONFIDENCE = 0.3

JOINT_NAMES = [
    "Nose", "Neck", "R.Shoulder", "R.Elbow", "R.Wrist",
    "L.Shoulder", "L.Elbow", "L.Wrist", "R.Hip", "R.Knee",
    "R.Ankle", "L.Hip", "L.Knee", "L.Ankle", "R.Eye", "L.Eye",
    "R.Ear", "L.Ear"
]
JOINT_WEIGHTS = np.array([
    0.8, 1.2, 1.0, 0.8, 0.5, 1.0, 0.8, 0.5,
    1.2, 0.9, 0.6, 1.2, 0.9, 0.6, 0.4, 0.4, 0.4, 0.4
], dtype=np.float32)

log = logging.getLogger(__name__)


# ------------------- Структуры данных -------------------
@dataclass
class ImageMeta:
    path: Path
    name: str
    size: int
    mtime: float
    pose: Optional[np.ndarray] = field(default=None, repr=False)


# ------------------- Логирование -------------------
def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


# ------------------- Работа с кешем поз -------------------
class PoseCache:
    """Загрузка и сохранение кеша нормализованных поз."""

    def __init__(self, cache_path: Optional[Path]):
        self.cache_path = cache_path
        self._cache: Dict[str, dict] = {}

    def load(self) -> Dict[str, dict]:
        """Возвращает словарь из файла кеша (или пустой)."""
        if not self.cache_path or not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get("_cache_version") != CACHE_VERSION:
                log.warning("Версия кеша устарела. Кеш будет пересоздан.")
                return {}
            self._cache = data
            return data
        except Exception as e:
            log.warning("Не удалось прочитать кеш: %s", e)
            return {}

    def save(self, items: List[ImageMeta]) -> None:
        """Сохраняет актуальные позы в файл."""
        if not self.cache_path:
            return
        try:
            serializable = {"_cache_version": CACHE_VERSION}
            for it in items:
                if it.pose is not None:
                    serializable[str(it.path)] = {
                        "pose": it.pose.tolist(),
                        "mtime": it.mtime,
                        "size": it.size
                    }
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            log.info("💾 Кеш поз сохранён: %s", self.cache_path)
        except Exception as e:
            log.warning("Ошибка сохранения кеша: %s", e)

    def get_entry(self, path: str) -> Optional[dict]:
        """Возвращает закешированную запись для пути."""
        return self._cache.get(path)


# ------------------- Экстрактор поз -------------------
class PoseExtractor:
    """Извлечение и нормализация поз с помощью DWPose, потокобезопасно."""

    def __init__(self, model: DwposeDetector):
        self.model = model
        self._lock = threading.Lock()

    @staticmethod
    def prepare_image(img: Image.Image, max_size: int = DWPOSE_RES) -> Image.Image:
        """EXIF-трансформация, конвертация в RGB, ресайз при необходимости."""
        img = ImageOps.exif_transpose(img)
        if img is None:
            raise ValueError("Не удалось применить EXIF-трансформацию")
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        return img

    def extract(self, image_path: Path) -> Optional[np.ndarray]:
        """Извлекает и нормализует позу, возвращает (18,3) массив или None."""
        try:
            log.debug("Обработка: %s", image_path.name)
            with Image.open(image_path) as img:
                img_prep = self.prepare_image(img)

            with self._lock:
                _, j, _ = self.model(
                    img_prep,
                    include_hand=False,
                    include_face=False,
                    include_body=True,
                    image_and_json=True,
                    detect_resolution=DWPOSE_RES
                )

            if not j or not j.get("people"):
                log.debug("Поза не обнаружена в %s", image_path.name)
                return None

            kp = j["people"][0].get("pose_keypoints_2d", [])
            pose = np.array(kp, dtype=np.float32).reshape(-1, 3)

            neck = pose[1]
            rhip, lhip = pose[8], pose[11]

            if neck[2] <= MIN_TORSO_CONFIDENCE:
                log.debug("Шея не распознана в %s", image_path.name)
                return None

            hip_points = [hip[:2] for hip in (rhip, lhip) if hip[2] > MIN_TORSO_CONFIDENCE]
            if not hip_points:
                log.debug("Бёдра не распознаны в %s", image_path.name)
                return None

            mid_hip = np.mean(hip_points, axis=0)
            center = (neck[:2] + mid_hip) / 2.0
            torso_len = np.linalg.norm(neck[:2] - mid_hip)

            if torso_len < 1e-4:
                return None

            pose[:, :2] = (pose[:, :2] - center) / torso_len
            return pose

        except UnidentifiedImageError:
            log.warning("⚠️ Файл не является изображением или повреждён: %s", image_path.name)
            return None
        except Exception as e:
            log.error("Ошибка обработки %s: %s", image_path.name, e)
            return None


# ------------------- Сбор файлов изображений -------------------
class ImageCollector:
    """Обходит директорию и возвращает список ImageMeta."""

    @staticmethod
    def collect(directory: Path, recursive: bool = False) -> List[ImageMeta]:
        pattern = "**/*" if recursive else "*"
        files = []
        for p in directory.glob(pattern):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    stat = p.stat()
                    files.append(ImageMeta(
                        path=p.resolve(),
                        name=p.name,
                        size=stat.st_size,
                        mtime=stat.st_mtime
                    ))
                except OSError as e:
                    log.debug("Не удалось получить метаданные %s: %s", p.name, e)
        return files


# ------------------- Поиск дубликатов -------------------
class DuplicateFinder:
    """Сравнение поз, формирование групп дубликатов."""

    def __init__(
            self,
            pose_extractor: PoseExtractor,
            cache: PoseCache,
            pose_threshold: float,
            max_joint_dist: float,
            min_common_joints: int,
            num_workers: int = 1
    ):
        self.extractor = pose_extractor
        self.cache = cache
        self.pose_threshold = pose_threshold
        self.max_joint_dist = max_joint_dist
        self.min_common_joints = min_common_joints
        self.num_workers = num_workers

    def find(self, files: List[ImageMeta]) -> List[List[ImageMeta]]:
        """Главный метод: извлечение поз и поиск групп."""
        # Загрузка кеша
        cache_data = self.cache.load()
        to_process = []

        for item in files:
            path_str = str(item.path)
            entry = cache_data.get(path_str) if cache_data else None
            if (entry and entry.get("mtime") == item.mtime and entry.get("size") == item.size):
                pose_list = entry.get("pose")
                item.pose = np.array(pose_list, dtype=np.float32) if pose_list else None
            else:
                to_process.append(item)

        # Извлечение поз для новых/изменённых файлов
        if to_process:
            log.info("Извлечение поз для %d изображений (потоков: %d)...", len(to_process), self.num_workers)
            if self.num_workers > 1:
                log.warning(
                    "⚠️ Многопоточность с GPU-моделями может вызывать ошибки CUDA. При сбоях используйте --workers 1")

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_item = {
                    executor.submit(self.extractor.extract, item.path): item
                    for item in to_process
                }
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        item.pose = future.result()
                    except Exception as e:
                        log.error("Поток завершился с ошибкой для %s: %s", item.path.name, e)
                        item.pose = None

            # Сохраняем обновлённый кеш
            self.cache.save(files)
        else:
            log.info("✅ Все позы загружены из актуального кеша.")

        # Отбираем валидные изображения
        valid_items = [it for it in files if it.pose is not None]
        if len(valid_items) < 2:
            return []

        # Попарное сравнение
        log.info("Попарное сравнение (порог RMSE=%.3f)...", self.pose_threshold)
        poses = np.array([it.pose for it in valid_items])
        conf_masks = poses[:, :, 2] > MIN_CONFIDENCE
        n = len(valid_items)
        graph = defaultdict(set)

        for i in range(n):
            p1, mask1 = poses[i], conf_masks[i]
            for j in range(i + 1, n):
                p2, mask2 = poses[j], conf_masks[j]
                common_mask = mask1 & mask2
                valid_count = int(common_mask.sum())

                if valid_count < self.min_common_joints:
                    continue

                diffs = p1[common_mask, :2] - p2[common_mask, :2]
                dists = np.linalg.norm(diffs, axis=1)
                max_dist = float(dists.max())

                if max_dist > self.max_joint_dist:
                    continue

                weights = JOINT_WEIGHTS[common_mask]
                rmse = float(np.sqrt(np.sum(weights * dists ** 2) / np.sum(weights)))

                if rmse <= self.pose_threshold:
                    path1, path2 = valid_items[i].path, valid_items[j].path
                    graph[path1].add(path2)
                    graph[path2].add(path1)

        # Формирование групп
        visited = set()
        groups = []
        info_by_path = {it.path: it for it in valid_items}

        for p in info_by_path:
            if p not in visited:
                stack, group = [p], []
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


# ------------------- Запись JSON-отчёта -------------------
class ReportWriter:
    """Генерация и сохранение JSON-отчёта."""

    @staticmethod
    def save(groups: List[List[ImageMeta]], output_path: Path) -> Path:
        data = []
        for i, group in enumerate(groups, 1):
            group.sort(key=lambda x: x.size, reverse=True)
            files_data = [{"path": str(f.path), "name": f.name, "size": f.size} for f in group]
            data.append({"group_id": i, "files": files_data})

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"groups": data}, f, ensure_ascii=False, indent=2)

        log.info("📄 JSON-отчёт сохранён: %s", output_path)
        return output_path


# ------------------- Главный класс приложения -------------------
class App:
    """Координирует загрузку, анализ и вывод результатов."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        setup_logging(args.debug)
        self.target_dir = Path(args.directory).resolve()
        self.model = self._load_model()
        self.pose_extractor = PoseExtractor(self.model)
        cache_path = None if args.no_cache else self.target_dir / POSE_CACHE_FILE
        self.cache = PoseCache(cache_path)
        self.finder = DuplicateFinder(
            pose_extractor=self.pose_extractor,
            cache=self.cache,
            pose_threshold=args.pose_threshold,
            max_joint_dist=args.max_joint_dist,
            min_common_joints=args.min_common_joints,
            num_workers=args.workers
        )

    @staticmethod
    def _load_model() -> DwposeDetector:
        log.info("Загрузка модели DWPose...")
        try:
            model = DwposeDetector.from_pretrained_default()
            log.info("✅ Модель успешно загружена.")
            return model
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить DWPose: {e}") from e

    def run(self) -> None:
        """Основной метод выполнения."""
        if not self.target_dir.is_dir():
            log.error("❌ Директория '%s' не найдена.", self.target_dir)
            sys.exit(1)

        files = ImageCollector.collect(self.target_dir, recursive=self.args.recursive)
        if not files:
            log.warning("⚠️ Поддерживаемые изображения не найдены.")
            sys.exit(0)

        log.info("📂 Найдено %d изображений.", len(files))

        try:
            groups = self.finder.find(files)
        except KeyboardInterrupt:
            log.warning("⛔ Прервано пользователем.")
            sys.exit(130)
        except Exception as e:
            log.critical("💥 Критическая ошибка: %s", e, exc_info=self.args.debug)
            sys.exit(1)

        if not groups:
            log.info("✅ Дубликаты по позе не найдены.")
            sys.exit(0)

        output_path = self.target_dir / self.args.output
        json_path = ReportWriter.save(groups, output_path)
        print(f"\n✅ Готово. JSON отчёт: {json_path}")
        print(f"💡 Чтобы посмотреть результат, откройте view_duplicates.html и загрузите {self.args.output}")


# ------------------- CLI -------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Поиск дубликатов изображений по позе (DWPose).")
    parser.add_argument("directory", nargs="?", default=".", help="Директория с изображениями.")
    parser.add_argument("--recursive", "-r", action="store_true", help="Сканировать подпапки.")
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
    parser.add_argument("--output", "-o", default=DEFAULT_JSON_REPORT, help="Имя JSON-отчёта.")
    return parser.parse_args()


if __name__ == "__main__":
    app = App(parse_args())
    app.run()