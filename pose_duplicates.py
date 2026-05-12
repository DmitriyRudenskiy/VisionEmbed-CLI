#!/usr/bin/env python3
"""
Поиск дубликатов изображений по позе человека с помощью DWPose (ООП-версия).
Нормализация по длине торса, взвешенный RMSE, ограничение на отклонение сустава.
Результат сохраняется в JSON файл.
"""
from __future__ import annotations

import os
import sys
import json
import logging
import argparse
import tempfile
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
        "Установите его или активируйте правильное окружение.",
        file=sys.stderr
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
CACHE_VERSION = 2
MIN_CONFIDENCE = 0.3
MIN_TORSO_CONFIDENCE = 0.3

# Вынесено для производительности
try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS

# COCO-18 формат
JOINT_WEIGHTS = np.array([
    0.8, 1.2, 1.0, 0.8, 0.5, 1.0, 0.8, 0.5,
    1.2, 0.9, 0.6, 1.2, 0.9, 0.6, 0.4, 0.4, 0.4, 0.4
], dtype=np.float32)

log = logging.getLogger(__name__)


# ------------------- Структуры данных -------------------
@dataclass(slots=True)
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
    """Загрузка и сохранение кеша нормализованных поз (относительные пути)."""

    def __init__(self, cache_path: Optional[Path], base_dir: Path):
        self.cache_path = cache_path
        self.base_dir = base_dir
        self._cache: Dict[str, dict] = {}

    def _to_relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.base_dir))
        except ValueError:
            return str(path)

    def load(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get("_cache_version") != CACHE_VERSION:
                log.warning("Версия кеша устарела или изменена структура. Кеш будет пересоздан.")
                return
            self._cache = data
            log.info("📦 Кеш загружен: %d записей.", len(data) - 1)
        except Exception as e:
            log.warning("Не удалось прочитать кеш: %s", e)

    def save(self, items: List[ImageMeta]) -> None:
        if not self.cache_path:
            return
        serializable = {"_cache_version": CACHE_VERSION}
        for it in items:
            if it.pose is not None:
                rel_path = self._to_relative(it.path)
                serializable[rel_path] = {
                    "pose": it.pose.tolist(),
                    "mtime": it.mtime,
                    "size": it.size
                }

        # Атомарная запись
        dir_name = self.cache_path.parent
        try:
            # Создаём временный файл в той же директории
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.json', text=True)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(serializable, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, self.cache_path)
                log.info("💾 Кеш поз сохранён: %s", self.cache_path)
            except:
                # При ошибке удаляем временный файл
                os.unlink(tmp_path)
                raise
        except Exception as e:
            log.warning("Ошибка сохранения кеша: %s", e)

    def get_valid_pose(self, item: ImageMeta) -> Optional[np.ndarray]:
        """Возвращает кешированную позу, если файл не изменился."""
        entry = self._cache.get(self._to_relative(item.path))
        if entry and entry.get("mtime") == item.mtime and entry.get("size") == item.size:
            pose_list = entry.get("pose")
            if pose_list and len(pose_list) == 54:
                return np.array(pose_list, dtype=np.float32).reshape(-1, 3)
        return None


# ------------------- Экстрактор поз -------------------
class PoseExtractor:
    """Извлечение и нормализация поз с помощью DWPose, потокобезопасно."""

    def __init__(self, model: DwposeDetector):
        self.model = model
        self._lock = threading.Lock()

    @staticmethod
    def prepare_image(img: Image.Image, max_size: int = DWPOSE_RES) -> Image.Image:
        img = ImageOps.exif_transpose(img) or img
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), _LANCZOS)
        return img

    def extract(self, image_path: Path) -> Optional[np.ndarray]:
        try:
            with Image.open(image_path) as img:
                img_prep = self.prepare_image(img)

            with self._lock:
                result = self.model(
                    img_prep,
                    include_hand=False,
                    include_face=False,
                    include_body=True,
                    image_and_json=True,
                    detect_resolution=DWPOSE_RES
                )

            if isinstance(result, tuple) and len(result) >= 2:
                j = result[1]
            elif isinstance(result, dict):
                j = result
            else:
                log.debug("Неожиданный формат ответа модели для %s", image_path.name)
                return None

            if not j or not j.get("people"):
                return None

            if len(j["people"]) > 1:
                log.debug("На фото %s найдено %d людей. Используется поза первого человека.",
                          image_path.name, len(j["people"]))

            kp = j["people"][0].get("pose_keypoints_2d", [])
            if not kp or len(kp) != 54:
                return None

            pose = np.array(kp, dtype=np.float32).reshape(-1, 3)
            return self._normalize_pose(pose, image_path.name)

        except UnidentifiedImageError:
            log.warning("⚠️ Файл не является изображением или повреждён: %s", image_path.name)
            return None
        except Exception as e:
            log.error("Ошибка обработки %s: %s", image_path.name, e)
            return None

    @staticmethod
    def _normalize_pose(pose: np.ndarray, filename: str) -> Optional[np.ndarray]:
        neck = pose[1]
        rhip, lhip = pose[8], pose[11]

        if neck[2] <= MIN_TORSO_CONFIDENCE:
            log.debug("Шея не распознана в %s", filename)
            return None

        hip_points = [hip[:2] for hip in (rhip, lhip) if hip[2] > MIN_TORSO_CONFIDENCE]
        if not hip_points:
            log.debug("Бёдра не распознаны в %s", filename)
            return None

        mid_hip = np.mean(hip_points, axis=0)
        center = (neck[:2] + mid_hip) / 2.0
        torso_len = np.linalg.norm(neck[:2] - mid_hip)

        if torso_len < 1e-4:
            return None

        pose[:, :2] = (pose[:, :2] - center) / torso_len
        return pose


# ------------------- Сбор файлов изображений -------------------
class ImageCollector:
    @staticmethod
    def collect(directory: Path, recursive: bool = False) -> List[ImageMeta]:
        files = []

        def _scan(d: Path):
            try:
                with os.scandir(d) as it:
                    for entry in it:
                        if entry.is_file(follow_symlinks=False):
                            p = Path(entry.path)  # entry.path уже абсолютный
                            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                                try:
                                    stat = entry.stat()
                                    files.append(ImageMeta(
                                        path=p,
                                        name=p.name,
                                        size=stat.st_size,
                                        mtime=stat.st_mtime
                                    ))
                                except OSError as e:
                                    log.debug("Не удалось получить метаданные %s: %s", p.name, e)
                        elif recursive and entry.is_dir(follow_symlinks=False):
                            _scan(Path(entry.path))
            except PermissionError:
                log.warning("Нет прав доступа к директории: %s", d)

        # Приводим корневую директорию к абсолютному пути
        _scan(directory.resolve())
        return files


# ------------------- Поиск дубликатов -------------------
class DuplicateFinder:
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
        self.cache.load()
        to_process: List[ImageMeta] = []

        # 1. Загрузка из кеша
        for item in files:
            item.pose = self.cache.get_valid_pose(item)
            if item.pose is None:
                to_process.append(item)

        # 2. Извлечение новых поз
        if to_process:
            log.info("Извлечение поз для %d изображений (потоков: %d)...", len(to_process), self.num_workers)
            if self.num_workers > 1:
                log.warning(
                    "⚠️ Многопоточность с GPU-моделями может вызывать ошибки CUDA. При сбоях используйте --workers 1")

            processed_count = 0
            try:
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

                        processed_count += 1
                        if processed_count % 50 == 0 or processed_count == len(to_process):
                            log.info("Обработано: %d/%d", processed_count, len(to_process))
            except KeyboardInterrupt:
                log.warning("⛔ Прервано пользователем во время извлечения поз.")
                executor.shutdown(wait=False)
                sys.exit(130)

            # Сохраняем кеш только после успешной обработки новых файлов
            self.cache.save(files)
        else:
            log.info("✅ Все позы загружены из актуального кеша.")

        valid_items = [it for it in files if it.pose is not None]
        if len(valid_items) < 2:
            log.info("Недостаточно изображений с валидными позами для сравнения.")
            return []

        # 3. Попарное сравнение
        log.info("Попарное сравнение (порог RMSE=%.3f)...", self.pose_threshold)
        groups = self._compare_and_group(valid_items)
        log.info("🎯 Найдено %d групп дубликатов.", len(groups))
        return groups

    def _compare_and_group(self, items: List[ImageMeta]) -> List[List[ImageMeta]]:
        poses = np.array([it.pose for it in items])
        poses_xy = poses[:, :, :2]
        conf_masks = poses[:, :, 2] > MIN_CONFIDENCE

        n = len(items)
        graph: Dict[Path, set] = defaultdict(set)

        max_joint_dist_sq = self.max_joint_dist ** 2
        total_comparisons = n * (n - 1) // 2
        log_interval = max(1, total_comparisons // 20)
        comparisons_done = 0

        for i in range(n):
            m1 = conf_masks[i]
            p1_xy = poses_xy[i]

            for j in range(i + 1, n):
                comparisons_done += 1
                if comparisons_done % log_interval == 0:
                    log.info(
                        f"Сравнение: {comparisons_done}/{total_comparisons} ({comparisons_done * 100 // total_comparisons}%)")

                common = m1 & conf_masks[j]
                valid_count = int(common.sum())

                if valid_count < self.min_common_joints:
                    continue

                diffs = p1_xy[common] - poses_xy[j, common]
                dists_sq = (diffs ** 2).sum(axis=1)

                if dists_sq.max() > max_joint_dist_sq:
                    continue

                w = JOINT_WEIGHTS[common]
                rmse = float(np.sqrt(np.dot(w, dists_sq) / w.sum()))

                if rmse <= self.pose_threshold:
                    p_i, p_j = items[i].path, items[j].path
                    graph[p_i].add(p_j)
                    graph[p_j].add(p_i)

        return self._find_connected_components(items, graph)

    @staticmethod
    def _find_connected_components(
            items: List[ImageMeta],
            graph: Dict[Path, set]
    ) -> List[List[ImageMeta]]:
        info_by_path = {it.path: it for it in items}
        visited = set()
        groups = []

        for p in info_by_path:
            if p in visited or p not in graph:
                continue

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

        return groups


# ------------------- Запись JSON-отчёта -------------------
class ReportWriter:
    @staticmethod
    def save(groups: List[List[ImageMeta]], output_path: Path) -> Path:
        data = []
        for i, group in enumerate(groups, 1):
            group.sort(key=lambda x: x.size, reverse=True)
            files_data = [
                {"path": str(f.path), "name": f.name, "size": f.size}
                for f in group
            ]
            data.append({"group_id": i, "count": len(files_data), "files": files_data})

        report = {
            "version": 1,
            "total_groups": len(data),
            "total_files": sum(g["count"] for g in data),
            "groups": data
        }

        # Атомарная запись отчета
        dir_name = output_path.parent
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.json', text=True)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, output_path)
            except:
                os.unlink(tmp_path)
                raise
        except Exception as e:
            log.error("Не удалось сохранить отчет атомарно: %s. Попытка прямой записи...", e)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

        log.info("📄 JSON-отчёт сохранён: %s", output_path)
        return output_path


# ------------------- Главный класс приложения -------------------
class App:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        setup_logging(args.debug)
        self.target_dir = Path(args.directory).resolve()
        self.model = self._load_model()
        self.pose_extractor = PoseExtractor(self.model)

        cache_path = None if args.no_cache else self.target_dir / POSE_CACHE_FILE
        self.cache = PoseCache(cache_path, self.target_dir)

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
            if hasattr(DwposeDetector, "from_pretrained_default"):
                model = DwposeDetector.from_pretrained_default()
            else:
                model = DwposeDetector()
            log.info("✅ Модель успешно загружена.")
            return model
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить DWPose: {e}") from e

    def run(self) -> None:
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
    parser = argparse.ArgumentParser(
        description="Поиск дубликатов изображений по позе человека (DWPose).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("directory", nargs="?", default=".", help="Директория с изображениями.")
    parser.add_argument("--recursive", "-r", action="store_true", help="Сканировать подпапки.")

    group = parser.add_argument_group("Параметры сравнения")
    group.add_argument("--pose-threshold", type=float, default=DEFAULT_POSE_THRESHOLD,
                       help="Порог взвешенного RMSE для признания дубликатом.")
    group.add_argument("--max-joint-dist", type=float, default=DEFAULT_MAX_JOINT_DIST,
                       help="Макс. отклонение одного сустава (после нормализации).")
    group.add_argument("--min-common-joints", type=int, default=DEFAULT_MIN_COMMON_JOINTS,
                       help="Мин. число общих уверенных точек для сравнения.")

    perf = parser.add_argument_group("Производительность")
    perf.add_argument("--workers", "-w", type=int, default=1,
                      help="Потоки для извлечения поз (рекомендуется 1 для GPU).")
    perf.add_argument("--no-cache", action="store_true", help="Отключить кеширование поз.")

    out = parser.add_argument_group("Вывод")
    out.add_argument("--output", "-o", default=DEFAULT_JSON_REPORT, help="Имя JSON-отчёта.")
    out.add_argument("--debug", action="store_true", help="Подробный вывод (DEBUG).")

    return parser.parse_args()


if __name__ == "__main__":
    app = App(parse_args())
    app.run()