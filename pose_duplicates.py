#!/usr/bin/env python3
"""
Поиск дубликатов изображений по позе человека с помощью DWPose (ООП-версия).
Нормализация по длине торса, взвешенный RMSE, ограничение на отклонение сустава.
Результат сохраняется в JSON файл.

Улучшения v2:
- Валидация входных параметров и структуры кеша
- torch.inference_mode() для ускорения и безопасности инференса
- Оптимизация векторных операций и маскирования
- Адаптивный прогресс-лог без спама
- Метаданные в JSON-отчёте (пороги, версия, дата)
- Graceful shutdown на всех этапах
- Строгая типизация и lazy-логирование
"""
from __future__ import annotations

import os
import sys
import json
import logging
import argparse
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError, ImageOps

log = logging.getLogger(__name__)

# ------------------- Импорт зависимостей -------------------
try:
    from dwpose import DwposeDetector
except ImportError:
    print(
        "❌ Ошибка: модуль 'dwpose' не найден. "
        "Установите его или активируйте правильное окружение.",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import torch

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    if hasattr(torch._C, "_set_print_stack_traces_on_fatal_signal"):
        torch._C._set_print_stack_traces_on_fatal_signal(False)
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ------------------- Константы -------------------
DEFAULT_POSE_THRESHOLD = 0.07
DEFAULT_MAX_JOINT_DIST = 0.12
DEFAULT_MIN_COMMON_JOINTS = 10
DWPOSE_RES = 1024
SUPPORTED_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
)
DEFAULT_JSON_REPORT = "duplicates.json"
POSE_CACHE_FILE = "pose_cache.json"
CACHE_VERSION = 3
MIN_CONFIDENCE = 0.3
MIN_TORSO_CONFIDENCE = 0.3

try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]

JOINT_WEIGHTS: NDArray[np.float32] = np.array(
    [
        0.8, 1.2, 1.0, 0.8, 0.5, 1.0, 0.8, 0.5,
        1.2, 0.9, 0.6, 1.2, 0.9, 0.6, 0.4, 0.4, 0.4, 0.4,
    ],
    dtype=np.float32,
)


# ------------------- Структуры данных -------------------
@dataclass(slots=True)
class ImageMeta:
    path: Path
    name: str
    size: int
    mtime: float
    pose: NDArray[np.float32] | None = field(default=None, repr=False)


# ------------------- Логирование -------------------
def setup_logging(debug: bool, quiet: bool = False) -> None:
    level = logging.DEBUG if debug else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log.setLevel(level)


# ------------------- Утилита: атомарная запись JSON -------------------
def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    dir_name = path.parent
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp.json", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        log.error("Атомарная запись не удалась: %s. Прямая запись...", e)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ------------------- Работа с кешем поз -------------------
class PoseCache:
    def __init__(self, cache_path: Path | None, base_dir: Path):
        self.cache_path = cache_path
        self.base_dir = base_dir
        self._cache: dict[str, dict[str, Any]] = {}
        self._dirty = False

    def _to_relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.base_dir))
        except ValueError:
            return str(path)

    def load(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
                log.warning("Версия кеша устарела или структура повреждена. Кеш будет пересоздан.")
                return
            entries = data.get("entries", {})
            if not isinstance(entries, dict):
                log.warning("Невалидная структура кеша. Пересоздание.")
                return
            self._cache = entries
            log.info("📦 Кеш загружен: %d записей.", len(self._cache))
        except Exception as e:
            log.warning("Не удалось прочитать кеш: %s", e)

    def save_if_dirty(self, processed_items: list[ImageMeta], all_files: list[ImageMeta]) -> None:
        if not self.cache_path or not self._dirty:
            return

        for it in processed_items:
            if it.pose is not None:
                rel = self._to_relative(it.path)
                self._cache[rel] = {
                    "pose": it.pose.tolist(),
                    "mtime": it.mtime,
                    "size": it.size,
                }

        all_paths = {self._to_relative(it.path) for it in all_files}
        stale = [k for k in self._cache if k not in all_paths]
        for k in stale:
            del self._cache[k]

        payload = {"version": CACHE_VERSION, "entries": self._cache}
        _atomic_write_json(self.cache_path, payload)
        self._dirty = False
        log.info("💾 Кеш поз обновлён: %s", self.cache_path)

    def get_valid_pose(self, item: ImageMeta) -> NDArray[np.float32] | None:
        entry = self._cache.get(self._to_relative(item.path))
        if not entry:
            return None
        if entry.get("mtime") != item.mtime or entry.get("size") != item.size:
            return None
        pose_list = entry.get("pose")
        if not isinstance(pose_list, list) or len(pose_list) != 54:
            return None
        return np.array(pose_list, dtype=np.float32).reshape(-1, 3)

    def mark_dirty(self) -> None:
        self._dirty = True


# ------------------- Экстрактор поз -------------------
class PoseExtractor:
    def __init__(self, model: DwposeDetector, device: str = "auto"):
        self.model = model
        self.device = device
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

    def extract(self, image_path: Path) -> NDArray[np.float32] | None:
        try:
            with Image.open(image_path) as img:
                img_prep = self.prepare_image(img)

            with self._lock:
                if _TORCH_AVAILABLE and hasattr(torch, "inference_mode"):
                    with torch.inference_mode():
                        result = self.model(
                            img_prep, include_hand=False, include_face=False,
                            include_body=True, image_and_json=True,
                            detect_resolution=DWPOSE_RES,
                        )
                else:
                    result = self.model(
                        img_prep, include_hand=False, include_face=False,
                        include_body=True, image_and_json=True,
                        detect_resolution=DWPOSE_RES,
                    )
            return self._parse_result(result, image_path.name)

        except UnidentifiedImageError:
            log.warning("⚠️ Файл не является изображением или повреждён: %s", image_path.name)
            return None
        except Exception as e:
            log.error("Ошибка обработки %s: %s", image_path.name, e)
            return None

    @staticmethod
    def _parse_result(result: object, filename: str) -> NDArray[np.float32] | None:
        j: dict[str, Any] | None = None
        if isinstance(result, tuple) and len(result) >= 2:
            j = result[1]
        elif isinstance(result, dict):
            j = result

        if not j or not j.get("people"):
            return None

        people = j["people"]
        if len(people) > 1:
            log.debug("На фото %s найдено %d людей. Используется поза первого.", filename, len(people))

        kp = people[0].get("pose_keypoints_2d", [])
        if not isinstance(kp, list) or len(kp) != 54:
            return None

        pose = np.array(kp, dtype=np.float32).reshape(-1, 3)
        return PoseExtractor._normalize_pose(pose, filename)

    @staticmethod
    def _normalize_pose(pose: NDArray[np.float32], filename: str) -> NDArray[np.float32] | None:
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
            log.debug("Длина торса слишком мала в %s", filename)
            return None

        norm_pose = pose.copy()
        norm_pose[:, :2] = (norm_pose[:, :2] - center) / torso_len
        return norm_pose


# ------------------- Сбор файлов изображений -------------------
class ImageCollector:
    @staticmethod
    def collect(directory: Path, recursive: bool = False) -> list[ImageMeta]:
        files: list[ImageMeta] = []

        def _scan(d: Path) -> None:
            try:
                with os.scandir(d) as it:
                    for entry in it:
                        if entry.is_file(follow_symlinks=False):
                            p = Path(entry.path)
                            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                                try:
                                    stat = entry.stat()
                                    files.append(ImageMeta(
                                        path=p, name=p.name, size=stat.st_size, mtime=stat.st_mtime
                                    ))
                                except OSError as e:
                                    log.debug("Не удалось получить метаданные %s: %s", p.name, e)
                        elif recursive and entry.is_dir(follow_symlinks=False):
                            _scan(Path(entry.path))
            except PermissionError:
                log.warning("Нет прав доступа к директории: %s", d)

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
            num_workers: int = 1,
    ):
        self.extractor = pose_extractor
        self.cache = cache
        self.pose_threshold = pose_threshold
        self.max_joint_dist = max_joint_dist
        self.min_common_joints = min_common_joints
        self.num_workers = num_workers

    def find(self, files: list[ImageMeta]) -> list[list[ImageMeta]]:
        self.cache.load()
        to_process: list[ImageMeta] = []

        for item in files:
            item.pose = self.cache.get_valid_pose(item)
            if item.pose is None:
                to_process.append(item)

        if to_process:
            try:
                self._extract_poses(to_process)
                self.cache.mark_dirty()
            finally:
                self.cache.save_if_dirty(
                    [it for it in to_process if it.pose is not None], files
                )
        else:
            log.info("✅ Все позы загружены из актуального кеша.")

        valid_items = [it for it in files if it.pose is not None]
        if len(valid_items) < 2:
            log.info("Недостаточно изображений с валидными позами для сравнения.")
            return []

        log.info(
            "Попарное сравнение %d изображений (порог RMSE=%.3f)...",
            len(valid_items), self.pose_threshold,
        )
        groups = self._compare_and_group(valid_items)
        log.info("🎯 Найдено %d групп дубликатов.", len(groups))
        return groups

    def _extract_poses(self, items: list[ImageMeta]) -> None:
        log.info("Извлечение поз для %d изображений (потоков: %d)...", len(items), self.num_workers)
        if self.num_workers > 1:
            log.warning(
                "⚠️ Многопоточность с GPU-моделями может вызывать ошибки CUDA. При сбоях используйте --workers 1")

        processed = 0
        total = len(items)
        log_interval = max(1, total // 20)

        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_item = {executor.submit(self.extractor.extract, item.path): item for item in items}
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        item.pose = future.result()
                    except Exception as e:
                        log.error("Поток завершился с ошибкой для %s: %s", item.path.name, e)
                        item.pose = None

                    processed += 1
                    if processed % log_interval == 0 or processed == total:
                        log.info("Обработано: %d/%d (%.0f%%)", processed, total, processed / total * 100)
        except KeyboardInterrupt:
            log.warning("⛔ Прервано пользователем во время извлечения поз.")
            raise

    def _compare_and_group(self, items: list[ImageMeta]) -> list[list[ImageMeta]]:
        n = len(items)
        if n < 2:
            return []

        poses = np.array([it.pose for it in items], dtype=np.float32)
        poses_xy = poses[:, :, :2]
        conf_masks = poses[:, :, 2] > MIN_CONFIDENCE

        edges: list[tuple[int, int]] = []
        max_joint_dist_sq = self.max_joint_dist ** 2
        weights = JOINT_WEIGHTS[np.newaxis, :]

        log_interval = max(1, (n - 1) // 20)

        try:
            for i in range(n - 1):
                if i % log_interval == 0:
                    log.info("Сравнение: строка %d/%d (~%.0f%%)", i + 1, n - 1, (i / max(n - 1, 1)) * 100)

                m1 = conf_masks[i]
                p1_xy = poses_xy[i]
                m_rest = conf_masks[i + 1:]
                p_rest_xy = poses_xy[i + 1:]

                common = m1[np.newaxis, :] & m_rest
                valid_counts = common.sum(axis=1)
                enough = valid_counts >= self.min_common_joints
                if not np.any(enough):
                    continue

                diffs = p1_xy[np.newaxis, :, :] - p_rest_xy
                dists_sq = np.einsum('ijk,ijk->ij', diffs, diffs)

                # Безопасный максимум: -inf для несовпадающих суставов
                max_dists = np.where(common, dists_sq, -np.inf).max(axis=1)

                weighted_sum = (weights * dists_sq * common).sum(axis=1)
                weight_total = (weights * common).sum(axis=1)
                rmse = np.sqrt(weighted_sum / np.maximum(weight_total, 1e-10))

                is_dup = enough & (max_dists <= max_joint_dist_sq) & (rmse <= self.pose_threshold)
                dup_indices = np.flatnonzero(is_dup) + i + 1
                edges.extend((i, int(j)) for j in dup_indices)
        except KeyboardInterrupt:
            log.warning("⛔ Прервано пользователем во время сравнения.")
            raise

        if not edges:
            return []

        graph: defaultdict[int, set[int]] = defaultdict(set)
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)

        visited = [False] * n
        groups: list[list[ImageMeta]] = []

        for start in range(n):
            if visited[start] or start not in graph:
                continue
            stack = [start]
            component: list[ImageMeta] = []
            while stack:
                v = stack.pop()
                if visited[v]:
                    continue
                visited[v] = True
                component.append(items[v])
                for nb in graph[v]:
                    if not visited[nb]:
                        stack.append(nb)
            if len(component) > 1:
                groups.append(component)

        return groups


# ------------------- Запись JSON-отчёта -------------------
class ReportWriter:
    @staticmethod
    def save(groups: list[list[ImageMeta]], output_path: Path, params: dict[str, Any]) -> Path:
        data = []
        for i, group in enumerate(groups, 1):
            group.sort(key=lambda x: x.size, reverse=True)
            files_data = [{"path": str(f.path), "name": f.name, "size": f.size} for f in group]
            data.append({"group_id": i, "count": len(files_data), "files": files_data})

        report = {
            "version": 2,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "params": params,
            "total_groups": len(data),
            "total_files": sum(g["count"] for g in data),
            "groups": data,
        }

        _atomic_write_json(output_path, report)
        log.info("📄 JSON-отчёт сохранён: %s", output_path)
        return output_path


# ------------------- Главный класс приложения -------------------
class App:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        setup_logging(args.debug, args.quiet)
        self.target_dir = Path(args.directory).resolve()
        self.model = self._load_model(args.device)
        self.pose_extractor = PoseExtractor(self.model, device=args.device)

        cache_path = None if args.no_cache else self.target_dir / POSE_CACHE_FILE
        self.cache = PoseCache(cache_path, self.target_dir)

        self.finder = DuplicateFinder(
            pose_extractor=self.pose_extractor,
            cache=self.cache,
            pose_threshold=args.pose_threshold,
            max_joint_dist=args.max_joint_dist,
            min_common_joints=args.min_common_joints,
            num_workers=args.workers,
        )

    @staticmethod
    def _load_model(device: str) -> DwposeDetector:
        log.info("Загрузка модели DWPose (device: %s)...", device)
        try:
            kwargs: dict[str, Any] = {}
            if device != "auto":
                kwargs["device"] = device

            if hasattr(DwposeDetector, "from_pretrained_default"):
                model = DwposeDetector.from_pretrained_default(**kwargs)
            else:
                model = DwposeDetector(**kwargs)

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
        params = {
            "pose_threshold": self.args.pose_threshold,
            "max_joint_dist": self.args.max_joint_dist,
            "min_common_joints": self.args.min_common_joints,
            "workers": self.args.workers,
            "recursive": self.args.recursive,
        }
        json_path = ReportWriter.save(groups, output_path, params)

        if not self.args.quiet:
            print(f"\n✅ Готово. JSON отчёт: {json_path}")
            print(f"💡 Чтобы посмотреть результат, откройте view_duplicates.html и загрузите {self.args.output}")
            total_dup_files = sum(len(g) for g in groups)
            print(f"📊 Найдено {len(groups)} групп, всего {total_dup_files} файлов.")


# ------------------- CLI -------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Поиск дубликатов изображений по позе человека (DWPose).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("directory", nargs="?", default=".", help="Директория с изображениями.")
    parser.add_argument("--recursive", "-r", action="store_true", help="Сканировать подпапки.")

    cmp = parser.add_argument_group("Параметры сравнения")
    cmp.add_argument("--pose-threshold", type=float, default=DEFAULT_POSE_THRESHOLD,
                     help="Порог взвешенного RMSE для признания дубликатом.")
    cmp.add_argument("--max-joint-dist", type=float, default=DEFAULT_MAX_JOINT_DIST,
                     help="Макс. отклонение одного сустава (после нормализации).")
    cmp.add_argument("--min-common-joints", type=int, default=DEFAULT_MIN_COMMON_JOINTS,
                     help="Мин. число общих уверенных точек для сравнения.")

    perf = parser.add_argument_group("Производительность")
    perf.add_argument("--workers", "-w", type=int, default=1,
                      help="Потоки для извлечения поз (рекомендуется 1 для GPU).")
    perf.add_argument("--no-cache", action="store_true", help="Отключить кеширование поз.")
    perf.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                      help="Устройство для инференса модели.")

    out = parser.add_argument_group("Вывод")
    out.add_argument("--output", "-o", default=DEFAULT_JSON_REPORT, help="Имя JSON-отчёта.")
    out.add_argument("--debug", action="store_true", help="Подробный вывод (DEBUG).")
    out.add_argument("--quiet", "-q", action="store_true", help="Минимальный вывод (только ошибки).")

    args = parser.parse_args()

    # Валидация параметров
    if args.workers < 1:
        parser.error("--workers должен быть ≥ 1")
    if args.pose_threshold <= 0:
        parser.error("--pose-threshold должен быть > 0")
    if args.max_joint_dist <= 0:
        parser.error("--max-joint-dist должен быть > 0")
    if not (1 <= args.min_common_joints <= 18):
        parser.error("--min-common-joints должен быть в диапазоне [1, 18]")

    return args


if __name__ == "__main__":
    app = App(parse_args())
    app.run()