#!/usr/bin/env python3
"""
Поиск дубликатов изображений по позе человека с помощью DWPose (ООП-версия v3).

Нормализация по длине торса, взвешенный RMSE, ограничение на отклонение сустава.
Результат сохраняется в JSON файл.

Ключевые улучшения v3:
- Чистый API кеша: update()/prune()/save() вместо save_if_dirty()
- DuplicateMatch + FindResult для типобезопасного хранения результатов
- Оценки сходства (RMSE, max_joint_dist) в JSON-отчёте
- Обработка CUDA OOM с автоматическим повтором на половинном разрешении
- Ранняя фильтрация кандидатов до дорогих вычислений в попарном сравнении
- ProgressTracker с ETA для адаптивного прогресс-лога
- Ленивая инициализация PyTorch — нет побочных эффектов на уровне модуля
- Константы NUM_KEYPOINTS / FLAT_KEYPOINTS вместо магических чисел
- CLI-опции --min-group-size, --cache-file
"""
from __future__ import annotations

import json
import logging
import os
import sys
import argparse
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError, ImageOps

__all__ = ["App", "DuplicateFinder", "PoseExtractor", "PoseCache", "ImageCollector"]

log = logging.getLogger(__name__)

# ─── Зависимости ────────────────────────────────────────────────────────────
try:
    from dwpose import DwposeDetector
except ImportError:
    print(
        "❌ Модуль 'dwpose' не найден. Установите или активируйте окружение.",
        file=sys.stderr,
    )
    sys.exit(1)

# ─── Константы ──────────────────────────────────────────────────────────────
NUM_KEYPOINTS = 18
FLAT_KEYPOINTS = NUM_KEYPOINTS * 3  # 54

DEFAULT_POSE_THRESHOLD = 0.07
DEFAULT_MAX_JOINT_DIST = 0.12
DEFAULT_MIN_COMMON_JOINTS = 10
DEFAULT_MIN_GROUP_SIZE = 2
DWPOSE_RES = 1024
SUPPORTED_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
)
DEFAULT_JSON_REPORT = "duplicates.json"
POSE_CACHE_FILE = "pose_cache.json"
CACHE_VERSION = 4
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


# ─── Ленивая инициализация PyTorch ──────────────────────────────────────────
_torch_initialized = False


def _init_torch() -> None:
    """Настройка PyTorch при первом использовании (без побочных эффектов на import)."""
    global _torch_initialized
    if _torch_initialized:
        return
    _torch_initialized = True
    try:
        import torch

        torch.set_grad_enabled(False)
        # Один поток CPU — избегаем конкуренции с ThreadPoolExecutor
        torch.set_num_threads(1)
        if hasattr(torch._C, "_set_print_stack_traces_on_fatal_signal"):
            torch._C._set_print_stack_traces_on_fatal_signal(False)
    except ImportError:
        pass


def _torch_inference_context():
    """Контекстный менеджер для инференса: inference_mode > no_grad > passthrough."""
    try:
        import torch

        if hasattr(torch, "inference_mode"):
            return torch.inference_mode()
        return torch.no_grad()
    except ImportError:
        return nullcontext()


# ─── Структуры данных ───────────────────────────────────────────────────────
@dataclass(slots=True)
class ImageMeta:
    """Метаданные изображения с опциональной нормализованной позой."""

    path: Path
    name: str
    size: int
    mtime: float
    pose: NDArray[np.float32] | None = field(default=None, repr=False)


class DuplicateMatch(NamedTuple):
    """Связь-дубликат с оценками сходства."""

    idx_a: int
    idx_b: int
    rmse: float
    max_joint_dist: float


@dataclass
class FindResult:
    """Полный результат поиска дубликатов."""

    groups: list[list[ImageMeta]]
    matches: list[DuplicateMatch]
    valid_items: list[ImageMeta]


# ─── Логирование ────────────────────────────────────────────────────────────
def setup_logging(debug: bool, quiet: bool = False) -> None:
    level = logging.DEBUG if debug else (logging.WARNING if quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log.setLevel(level)


class ProgressTracker:
    """Адаптивный прогресс-лог с ETA."""

    def __init__(
        self, total: int, label: str = "Обработка", interval_pct: float = 5.0
    ):
        self.total = total
        self.label = label
        self._interval = interval_pct
        self._done = 0
        self._last_pct = -1.0
        self._start = time.monotonic()

    def advance(self, n: int = 1) -> None:
        self._done += n
        pct = (self._done / self.total * 100) if self.total else 100.0
        if (pct - self._last_pct >= self._interval) or self._done == self.total:
            self._last_pct = pct
            elapsed = time.monotonic() - self._start
            if 0 < self._done < self.total:
                eta = elapsed / self._done * (self.total - self._done)
                eta_str = f", ETA {eta:.0f}s" if eta >= 1 else ""
            else:
                eta_str = ""
            log.info(
                "%s: %d/%d (%.1f%%%s)", self.label, self._done, self.total, pct, eta_str
            )


# ─── Атомарная запись JSON ──────────────────────────────────────────────────
def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Атомарная запись JSON через временный файл + os.replace."""
    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, suffix=".tmp.json", text=True
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        log.error("Атомарная запись не удалась: %s — прямая запись", e)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ─── Кеш поз ────────────────────────────────────────────────────────────────
class PoseCache:
    """Кеш извлечённых поз с автоочисткой устаревших записей.

    API: load() → get_valid_pose() × N → update() × K → prune() → save().
    """

    def __init__(self, cache_path: Path | None, base_dir: Path):
        self.cache_path = cache_path
        self.base_dir = base_dir
        self._entries: dict[str, dict[str, Any]] = {}
        self._dirty = False

    def _to_relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.base_dir))
        except ValueError:
            return str(path)

    # ── Чтение ──

    def load(self) -> None:
        """Загрузить кеш. Несовместимая версия → пересоздание."""
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
                log.warning("Версия кеша устарела — пересоздание.")
                return
            entries = data.get("entries")
            if not isinstance(entries, dict):
                log.warning("Невалидная структура кеша — пересоздание.")
                return
            self._entries = entries
            log.info("📦 Кеш загружен: %d записей.", len(self._entries))
        except Exception as e:
            log.warning("Не удалось прочитать кеш: %s", e)

    def get_valid_pose(self, item: ImageMeta) -> NDArray[np.float32] | None:
        """Вернуть позу из кеша, если файл не изменился."""
        entry = self._entries.get(self._to_relative(item.path))
        if not entry:
            return None
        if entry.get("mtime") != item.mtime or entry.get("size") != item.size:
            return None
        pose_list = entry.get("pose")
        if not isinstance(pose_list, list) or len(pose_list) != FLAT_KEYPOINTS:
            return None
        return np.array(pose_list, dtype=np.float32).reshape(-1, 3)

    # ── Запись ──

    def update(self, item: ImageMeta) -> None:
        """Сохранить извлечённую позу в кеш (отмечает кеш как грязный)."""
        if item.pose is None:
            return
        rel = self._to_relative(item.path)
        self._entries[rel] = {
            "pose": item.pose.tolist(),
            "mtime": item.mtime,
            "size": item.size,
        }
        self._dirty = True

    def prune(self, existing_paths: set[str]) -> int:
        """Удалить записи для файлов, которых больше нет. Вернуть кол-во удалённых."""
        stale = [k for k in self._entries if k not in existing_paths]
        for k in stale:
            del self._entries[k]
        if stale:
            self._dirty = True
            log.debug("Удалено %d устаревших записей кеша.", len(stale))
        return len(stale)

    def save(self) -> None:
        """Записать кеш на диск, если были изменения."""
        if not self.cache_path or not self._dirty:
            return
        _atomic_write_json(
            self.cache_path, {"version": CACHE_VERSION, "entries": self._entries}
        )
        self._dirty = False
        log.info("💾 Кеш обновлён: %s (%d записей)", self.cache_path, len(self._entries))


# ─── Экстрактор поз ────────────────────────────────────────────────────────
class PoseExtractor:
    """Извлечение и нормализация поз с помощью DWPose.

    GPU-инференс сериализован через _inference_lock.
    При CUDA OOM — автоматический повтор на половинном разрешении.
    """

    def __init__(self, model: DwposeDetector, device: str = "auto"):
        self.model = model
        self.device = device
        self._inference_lock = threading.Lock()

    @staticmethod
    def prepare_image(img: Image.Image, max_size: int = DWPOSE_RES) -> Image.Image:
        """EXIF-трансформация, конвертация в RGB, ресайз."""
        img = ImageOps.exif_transpose(img) or img
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), _LANCZOS)
        return img

    def _run_inference(self, img: Image.Image) -> Any:
        """Потокобезопасный инференс с torch.inference_mode()."""
        with self._inference_lock:
            with _torch_inference_context():
                return self.model(
                    img,
                    include_hand=False,
                    include_face=False,
                    include_body=True,
                    image_and_json=True,
                    detect_resolution=DWPOSE_RES,
                )

    def extract(self, image_path: Path) -> NDArray[np.float32] | None:
        """Извлечь нормализованную позу из изображения."""
        try:
            with Image.open(image_path) as img:
                img_prep = self.prepare_image(img)
            result = self._run_inference(img_prep)
            return self._parse_result(result, image_path.name)
        except UnidentifiedImageError:
            log.warning("⚠️ Файл не распознан как изображение: %s", image_path.name)
            return None
        except Exception as e:
            if _is_oom_error(e):
                return self._retry_half_res(image_path)
            log.error("Ошибка обработки %s: %s", image_path.name, e)
            return None

    def _retry_half_res(self, image_path: Path) -> NDArray[np.float32] | None:
        """Повторная попытка с DWPOSE_RES/2 при CUDA OOM."""
        log.warning("⚠️ CUDA OOM: %s — повтор на %dpx", image_path.name, DWPOSE_RES // 2)
        _torch_clear_cache()
        try:
            with Image.open(image_path) as img:
                img_prep = self.prepare_image(img, max_size=DWPOSE_RES // 2)
            result = self._run_inference(img_prep)
            return self._parse_result(result, image_path.name)
        except Exception as e2:
            log.error("Повторная ошибка для %s: %s", image_path.name, e2)
            return None

    @staticmethod
    def _parse_result(result: Any, filename: str) -> NDArray[np.float32] | None:
        """Разбор ответа DWPose → массив (18, 3) или None."""
        j: dict[str, Any] | None = None
        if isinstance(result, tuple) and len(result) >= 2:
            j = result[1]
        elif isinstance(result, dict):
            j = result

        if not j or not j.get("people"):
            return None

        people = j["people"]
        if len(people) > 1:
            log.debug(
                "%s: %d людей — используется поза первого.", filename, len(people)
            )

        kp = people[0].get("pose_keypoints_2d", [])
        if not isinstance(kp, list) or len(kp) != FLAT_KEYPOINTS:
            return None

        pose = np.array(kp, dtype=np.float32).reshape(-1, 3)
        return PoseExtractor._normalize_pose(pose, filename)

    @staticmethod
    def _normalize_pose(
        pose: NDArray[np.float32], filename: str
    ) -> NDArray[np.float32] | None:
        """Нормализация: центр = середина торса, масштаб = длина торса."""
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
            log.debug("Длина торса ≈ 0 в %s", filename)
            return None

        norm_pose = pose.copy()
        norm_pose[:, :2] = (norm_pose[:, :2] - center) / torso_len
        return norm_pose


def _is_oom_error(error: Exception) -> bool:
    """Проверить, является ли ошибка нехваткой GPU-памяти."""
    msg = str(error).lower()
    return "out of memory" in msg or "cuda" in msg and "memory" in msg


def _torch_clear_cache() -> None:
    """Очистить CUDA-кеш при OOM."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ─── Сбор файлов ────────────────────────────────────────────────────────────
class ImageCollector:
    """Рекурсивный сбор поддерживаемых изображений из директории."""

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
                                    files.append(
                                        ImageMeta(
                                            path=p,
                                            name=p.name,
                                            size=stat.st_size,
                                            mtime=stat.st_mtime,
                                        )
                                    )
                                except OSError as e:
                                    log.debug(
                                        "Метаданные недоступны %s: %s", p.name, e
                                    )
                        elif recursive and entry.is_dir(follow_symlinks=False):
                            _scan(Path(entry.path))
            except PermissionError:
                log.warning("Нет прав доступа: %s", d)

        _scan(directory.resolve())
        return files


# ─── Поиск дубликатов ───────────────────────────────────────────────────────
class DuplicateFinder:
    """Попарное векторизованное сравнение поз + группировка в связные компоненты."""

    def __init__(
        self,
        pose_extractor: PoseExtractor,
        cache: PoseCache,
        pose_threshold: float,
        max_joint_dist: float,
        min_common_joints: int,
        min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
        num_workers: int = 1,
    ):
        self.extractor = pose_extractor
        self.cache = cache
        self.pose_threshold = pose_threshold
        self.max_joint_dist = max_joint_dist
        self.min_common_joints = min_common_joints
        self.min_group_size = min_group_size
        self.num_workers = num_workers

    def find(self, files: list[ImageMeta]) -> FindResult:
        """Основной пайплайн: кеш → извлечение → сравнение → группировка."""
        self.cache.load()
        to_process: list[ImageMeta] = []

        for item in files:
            item.pose = self.cache.get_valid_pose(item)
            if item.pose is None:
                to_process.append(item)

        if to_process:
            try:
                self._extract_poses(to_process)
            finally:
                # Гарантированное сохранение кеша даже при ошибке
                for it in to_process:
                    self.cache.update(it)
                existing = {self.cache._to_relative(it.path) for it in files}
                self.cache.prune(existing)
                self.cache.save()
        else:
            log.info("✅ Все позы загружены из актуального кеша.")

        valid_items = [it for it in files if it.pose is not None]
        if len(valid_items) < 2:
            log.info("Недостаточно изображений с валидными позами для сравнения.")
            return FindResult(groups=[], matches=[], valid_items=valid_items)

        log.info(
            "Попарное сравнение %d изображений (RMSE≤%.3f, max_joint≤%.3f)...",
            len(valid_items),
            self.pose_threshold,
            self.max_joint_dist,
        )

        matches = self._find_matches(valid_items)
        groups = self._build_groups(valid_items, matches)

        log.info("🎯 Найдено %d групп дубликатов.", len(groups))
        return FindResult(groups=groups, matches=matches, valid_items=valid_items)

    # ── Извлечение ──

    def _extract_poses(self, items: list[ImageMeta]) -> None:
        """I/O параллельно (ThreadPool), инференс сериализован (_inference_lock)."""
        log.info(
            "Извлечение поз: %d изображений (I/O-потоков: %d)...",
            len(items),
            self.num_workers,
        )
        progress = ProgressTracker(len(items), "Извлечение поз")

        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_item = {
                    executor.submit(self.extractor.extract, item.path): item
                    for item in items
                }
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        item.pose = future.result()
                    except Exception as e:
                        log.error("Ошибка потока для %s: %s", item.path.name, e)
                        item.pose = None
                    progress.advance()
        except KeyboardInterrupt:
            log.warning("⛔ Прервано при извлечении поз.")
            raise

    # ── Сравнение ──

    def _find_matches(self, items: list[ImageMeta]) -> list[DuplicateMatch]:
        """Векторизованное попарное сравнение с ранней фильтрацией кандидатов.

        Оптимизация: для каждой строки i вычисляем dists_sq только для пар,
        прошедших порог min_common_joints, а не для всех (n-i) последующих.
        """
        n = len(items)
        poses = np.array([it.pose for it in items], dtype=np.float32)  # (N, 18, 3)
        poses_xy = poses[:, :, :2]  # (N, 18, 2)
        conf_masks = poses[:, :, 2] > MIN_CONFIDENCE  # (N, 18)

        matches: list[DuplicateMatch] = []
        max_dist_sq = np.float32(self.max_joint_dist ** 2)
        weights = JOINT_WEIGHTS[np.newaxis, :]  # (1, 18)

        progress = ProgressTracker(n - 1, "Сравнение")

        try:
            for i in range(n - 1):
                progress.advance()

                m1 = conf_masks[i]  # (18,)
                p1_xy = poses_xy[i]  # (18, 2) <- ИСПРАВЛЕНО: добавлена пропущенная строка

                # Общие уверенные суставы со всеми последующими строками
                common = m1[np.newaxis, :] & conf_masks[i + 1 :]  # (M, 18)
                valid_counts = common.sum(axis=1)  # (M,)

                # ★ Ранняя фильтрация — работаем только с кандидатами
                candidates = np.flatnonzero(valid_counts >= self.min_common_joints)
                if candidates.size == 0:
                    continue

                # Абсолютные индексы строк для fancy indexing (одна операция вместо среза)
                row_idx = candidates + (i + 1)
                p_cand = poses_xy[row_idx]  # (K, 18, 2)
                m_cand = common[candidates]  # (K, 18)

                diffs = p1_xy[np.newaxis, :, :] - p_cand  # (K, 18, 2)
                dists_sq = np.einsum("ijk,ijk->ij", diffs, diffs)  # (K, 18)

                # Макс. расстояние среди общих суставов (−inf для необщих)
                max_dists = np.where(m_cand, dists_sq, np.float32(-np.inf)).max(
                    axis=1
                )

                # Взвешенный RMSE
                weighted_sum = (weights * dists_sq * m_cand).sum(axis=1)
                weight_total = (weights * m_cand).sum(axis=1)
                rmse = np.sqrt(weighted_sum / np.maximum(weight_total, 1e-10))

                # Итоговая фильтрация
                is_dup = (max_dists <= max_dist_sq) & (rmse <= self.pose_threshold)
                dup_k = np.flatnonzero(is_dup)

                for k in dup_k:
                    j = int(row_idx[k])
                    matches.append(
                        DuplicateMatch(
                            idx_a=i,
                            idx_b=j,
                            rmse=float(rmse[k]),
                            max_joint_dist=float(np.sqrt(max_dists[k])),
                        )
                    )
        except KeyboardInterrupt:
            log.warning("⛔ Прервано при сравнении.")
            raise

        return matches

    # ── Группировка ──

    @staticmethod
    def _build_groups(
        items: list[ImageMeta], matches: list[DuplicateMatch]
    ) -> list[list[ImageMeta]]:
        """Связные компоненты графа совпадений."""
        if not matches:
            return []

        graph: defaultdict[int, set[int]] = defaultdict(set)
        for m in matches:
            graph[m.idx_a].add(m.idx_b)
            graph[m.idx_b].add(m.idx_a)

        n = len(items)
        visited = [False] * n
        groups: list[list[ImageMeta]] = []

        for start in range(n):
            if visited[start] or start not in graph:
                continue
            component: list[ImageMeta] = []
            stack = [start]
            while stack:
                v = stack.pop()
                if visited[v]:
                    continue
                visited[v] = True
                component.append(items[v])
                for nb in graph[v]:
                    if not visited[nb]:
                        stack.append(nb)
            if len(component) >= 2:
                groups.append(component)

        return groups


# ─── Отчёт ──────────────────────────────────────────────────────────────────
class ReportWriter:
    """JSON-отчёт с группами дубликатов и попарными оценками сходства."""

    @staticmethod
    def save(result: FindResult, output_path: Path, params: dict[str, Any]) -> Path:
        path_to_idx: dict[str, int] = {}
        idx_to_path: dict[int, str] = {}
        for i, it in enumerate(result.valid_items):
            path_to_idx[str(it.path)] = i
            idx_to_path[i] = str(it.path)

        # Индекс сходства по упорядоченной паре (min, max)
        pair_sim: dict[tuple[int, int], DuplicateMatch] = {}
        for m in result.matches:
            key = (min(m.idx_a, m.idx_b), max(m.idx_a, m.idx_b))
            if key not in pair_sim or m.rmse < pair_sim[key].rmse:
                pair_sim[key] = m

        data = []
        for gi, group in enumerate(result.groups, 1):
            group.sort(key=lambda x: x.size, reverse=True)
            files_data = [
                {"path": str(f.path), "name": f.name, "size": f.size} for f in group
            ]

            # Попарные оценки внутри группы
            group_indices = sorted(
                path_to_idx[str(f.path)] for f in group if str(f.path) in path_to_idx
            )
            pairs = []
            for ai in range(len(group_indices)):
                for bi in range(ai + 1, len(group_indices)):
                    key = (group_indices[ai], group_indices[bi])
                    if key in pair_sim:
                        m = pair_sim[key]
                        pairs.append(
                            {
                                "file_a": idx_to_path[m.idx_a],
                                "file_b": idx_to_path[m.idx_b],
                                "rmse": round(m.rmse, 4),
                                "max_joint_dist": round(m.max_joint_dist, 4),
                            }
                        )

            data.append(
                {"group_id": gi, "count": len(files_data), "files": files_data, "pairs": pairs}
            )

        report = {
            "version": 3,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "params": params,
            "total_groups": len(data),
            "total_files": sum(g["count"] for g in data),
            "groups": data,
        }

        _atomic_write_json(output_path, report)
        log.info("📄 JSON-отчёт сохранён: %s", output_path)
        return output_path


# ─── Приложение ─────────────────────────────────────────────────────────────
class App:
    """Главный класс CLI-приложения."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        setup_logging(args.debug, args.quiet)
        _init_torch()

        self.target_dir = Path(args.directory).resolve()
        self.model = self._load_model(args.device)
        self.pose_extractor = PoseExtractor(self.model, device=args.device)

        if args.no_cache:
            cache_path = None
        else:
            cf = Path(args.cache_file)
            cache_path = cf if cf.is_absolute() else self.target_dir / cf
        self.cache = PoseCache(cache_path, self.target_dir)

        self.finder = DuplicateFinder(
            pose_extractor=self.pose_extractor,
            cache=self.cache,
            pose_threshold=args.pose_threshold,
            max_joint_dist=args.max_joint_dist,
            min_common_joints=args.min_common_joints,
            min_group_size=args.min_group_size,
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
            log.info("✅ Модель загружена.")
            return model
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить DWPose: {e}") from e

    def run(self) -> None:
        if not self.target_dir.is_dir():
            log.error("❌ Директория не найдена: %s", self.target_dir)
            sys.exit(1)

        files = ImageCollector.collect(self.target_dir, recursive=self.args.recursive)
        if not files:
            log.warning("⚠️ Поддерживаемые изображения не найдены.")
            sys.exit(0)

        log.info("📂 Найдено %d изображений.", len(files))

        try:
            result = self.finder.find(files)
        except KeyboardInterrupt:
            log.warning("⛔ Прервано пользователем.")
            sys.exit(130)
        except Exception as e:
            log.critical("💥 Критическая ошибка: %s", e, exc_info=self.args.debug)
            sys.exit(1)

        if not result.groups:
            log.info("✅ Дубликаты по позе не найдены.")
            sys.exit(0)

        # Фильтрация по минимальному размеру группы
        if self.args.min_group_size > 2:
            result.groups = [g for g in result.groups if len(g) >= self.args.min_group_size]
            if not result.groups:
                log.info(
                    "✅ Нет групп размером ≥ %d (попробуйте --min-group-size 2).",
                    self.args.min_group_size,
                )
                sys.exit(0)

        output_path = self.target_dir / self.args.output
        params = {
            "pose_threshold": self.args.pose_threshold,
            "max_joint_dist": self.args.max_joint_dist,
            "min_common_joints": self.args.min_common_joints,
            "min_group_size": self.args.min_group_size,
            "workers": self.args.workers,
            "recursive": self.args.recursive,
        }
        json_path = ReportWriter.save(result, output_path, params)

        if not self.args.quiet:
            total_dup = sum(len(g) for g in result.groups)
            print(f"\n✅ Готово. JSON: {json_path}")
            print(f"📊 Найдено {len(result.groups)} групп, всего {total_dup} файлов.")


# ─── CLI ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Поиск дубликатов изображений по позе человека (DWPose).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "directory", nargs="?", default=".", help="Директория с изображениями."
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true", help="Сканировать подпапки."
    )

    cmp = parser.add_argument_group("Параметры сравнения")
    cmp.add_argument(
        "--pose-threshold",
        type=float,
        default=DEFAULT_POSE_THRESHOLD,
        help="Порог взвешенного RMSE.",
    )
    cmp.add_argument(
        "--max-joint-dist",
        type=float,
        default=DEFAULT_MAX_JOINT_DIST,
        help="Макс. отклонение одного сустава (после нормализации).",
    )
    cmp.add_argument(
        "--min-common-joints",
        type=int,
        default=DEFAULT_MIN_COMMON_JOINTS,
        help="Мин. число общих уверенных точек.",
    )
    cmp.add_argument(
        "--min-group-size",
        type=int,
        default=DEFAULT_MIN_GROUP_SIZE,
        help="Мин. размер группы для включения в отчёт.",
    )

    perf = parser.add_argument_group("Производительность")
    perf.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="I/O-потоки при извлечении (GPU-инференс всегда сериализован).",
    )
    perf.add_argument(
        "--no-cache", action="store_true", help="Отключить кеш поз."
    )
    perf.add_argument(
        "--cache-file",
        default=POSE_CACHE_FILE,
        help="Файл кеша (относительно целевой директории или абсолютный путь).",
    )
    perf.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Устройство инференса.",
    )

    out = parser.add_argument_group("Вывод")
    out.add_argument(
        "--output", "-o", default=DEFAULT_JSON_REPORT, help="Имя JSON-отчёта."
    )
    out.add_argument(
        "--debug", action="store_true", help="Подробный вывод (DEBUG)."
    )
    out.add_argument(
        "--quiet", "-q", action="store_true", help="Минимальный вывод (только ошибки)."
    )

    args = parser.parse_args()

    # Валидация параметров
    if args.workers < 1:
        parser.error("--workers должен быть ≥ 1")
    if args.pose_threshold <= 0:
        parser.error("--pose-threshold должен быть > 0")
    if args.max_joint_dist <= 0:
        parser.error("--max-joint-dist должен быть > 0")
    if not (1 <= args.min_common_joints <= NUM_KEYPOINTS):
        parser.error(f"--min-common-joints должен быть в диапазоне [1, {NUM_KEYPOINTS}]")
    if args.min_group_size < 2:
        parser.error("--min-group-size должен быть ≥ 2")

    return args


if __name__ == "__main__":
    app = App(parse_args())
    app.run()