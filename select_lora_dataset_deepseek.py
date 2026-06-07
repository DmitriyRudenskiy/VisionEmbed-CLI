#!/usr/bin/env python3
"""
CLI-утилита для подготовки датасета LoRA (FLUX) и создания benchmark-отчёта.
Версия: 3.4 (Robust Tensor Extraction)
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Конфигурация логирования
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("dataset_prep")

for noisy in ["transformers", "huggingface_hub", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

UTILITY_VERSION = "3.4 (Robust Tensor Extraction)"

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    """Абсолютный детерминизм."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def validate_filename(filename: str) -> bool:
    """
    Проверка, что имя файла соответствует требованиям:
    - только basename (без слешей)
    - не начинается с синтетических префиксов item_, file_, img_
    """
    if "/" in filename or "\\" in filename:
        return False
    if re.match(r'^(item_\d+|file_\d+|img_\d+)', filename, re.IGNORECASE):
        return False
    return True


def find_images(input_dir: str) -> list[Path]:
    """
    Поиск изображений ТОЛЬКО в указанной директории (без подпапок).
    Жёсткая лексикографическая сортировка абсолютных путей.
    Проверка имени на соответствие требованиям.
    """
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    paths = []
    input_path = Path(input_dir)
    for item in sorted(input_path.iterdir(), key=lambda p: p.name):
        if item.is_file() and item.suffix.lower() in extensions:
            if not validate_filename(item.name):
                logger.error(
                    f"❌ Запрещённое имя файла: {item.name}. "
                    f"Использование синтетических имён (item_*, file_*, img_*) или путей запрещено."
                )
                sys.exit(1)
            paths.append(item.resolve())
    return paths


def load_and_filter_images(
    paths: list[Path], min_size: int
) -> tuple[list[Path], list[int], int]:
    """Фильтрация: читаемость, размер. Возвращает валидные пути, исходные индексы и счётчик битых."""
    valid_paths = []
    valid_indices = []
    broken = 0
    for idx, p in enumerate(tqdm(paths, desc="Фильтрация изображений", leave=False)):
        try:
            with Image.open(p) as img:
                w, h = img.size
                if min(w, h) < min_size:
                    continue
                img = img.convert("RGB")
                _ = img
            valid_paths.append(p)
            valid_indices.append(idx)
        except Exception:
            broken += 1
    return valid_paths, valid_indices, broken


def compute_embeddings(
    model: CLIPModel,
    processor: CLIPProcessor,
    paths: list[Path],
    batch_size: int,
) -> np.ndarray:
    """Инференс строго на CPU, L2-нормализация. Прогресс показывает количество изображений."""
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    all_embeddings = []

    pbar = tqdm(total=len(paths), desc="Векторизация (CPU)", unit="img", leave=False)
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            outputs = model.get_image_features(pixel_values=pixel_values)
            if isinstance(outputs, torch.Tensor):
                embeddings_batch = outputs
            elif hasattr(outputs, "pooler_output"):
                embeddings_batch = outputs.pooler_output
            elif hasattr(outputs, "image_embeds"):
                embeddings_batch = outputs.image_embeds
            else:
                raise ValueError(
                    f"Неожиданный тип результата get_image_features: {type(outputs)}"
                )
        embeddings_batch = F.normalize(embeddings_batch, p=2, dim=1)
        all_embeddings.append(embeddings_batch.cpu().numpy().astype(np.float32))
        for img in images:
            img.close()
        pbar.update(len(batch_paths))
    pbar.close()
    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# Union-Find для дедупликации
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]


def deduplicate_chunked(
    embeddings: np.ndarray,
    file_paths: list[Path],
    original_indices: list[int],
    threshold: float,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, list[Path], list[int], int]:
    """Чанкованная дедупликация на основе косинусного сходства."""
    n = embeddings.shape[0]
    uf = UnionFind(n)

    for i in tqdm(range(0, n, chunk_size), desc="Дедупликация (чанки)", leave=False):
        end_i = min(i + chunk_size, n)
        chunk_a = embeddings[i:end_i]
        for j in range(0, n, chunk_size):
            end_j = min(j + chunk_size, n)
            chunk_b = embeddings[j:end_j]
            sim = chunk_a @ chunk_b.T
            rows, cols = np.where(sim >= threshold)
            for r, c in zip(rows, cols):
                global_r = i + r
                global_c = j + c
                if global_r == global_c:
                    continue
                uf.union(global_r, global_c)

    clusters = {}
    for idx in range(n):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)

    representatives = []
    removed_count = 0
    for comp in clusters.values():
        best_idx = max(
            comp,
            key=lambda i: (file_paths[i].stat().st_size, -original_indices[i]),
        )
        representatives.append(best_idx)
        removed_count += len(comp) - 1

    representatives.sort(key=lambda i: original_indices[i])
    unique_embeddings = embeddings[representatives]
    unique_paths = [file_paths[i] for i in representatives]
    unique_orig_indices = [original_indices[i] for i in representatives]
    return unique_embeddings, unique_paths, unique_orig_indices, removed_count


# ---------------------------------------------------------------------------
# Кластеризация и очистка
# ---------------------------------------------------------------------------
def optimal_kmeans(
    embeddings: np.ndarray,
    max_clusters: int,
    seed: int,
) -> tuple[np.ndarray, int, float | None]:
    """Автоподбор k через силуэт."""
    n = embeddings.shape[0]
    upper = min(max_clusters, n // 2)
    if n < 4:
        k = min(n, 2)
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        return labels, k, None

    best_k, best_score, best_labels = 2, -1, None
    for k in range(2, upper + 1):
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels
    return best_labels, best_k, best_score


def filter_noise_and_outliers(
    embeddings: np.ndarray,
    labels: np.ndarray,
    paths: list[Path],
    orig_indices: list[int],
    min_cluster_size: int,
    outlier_percentile: float,
) -> tuple[np.ndarray, list[Path], list[int], np.ndarray, int, int]:
    """Шаги 5-6-6.1: удаление микро-кластеров и выбросов."""
    noise_removed_total = 0
    outlier_removed_total = 0

    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_clusters = set(
        lbl for lbl, cnt in zip(unique_labels, counts) if cnt >= min_cluster_size
    )
    noise_mask = np.isin(labels, list(valid_clusters), invert=True)
    noise_removed = noise_mask.sum()
    keep_mask = ~noise_mask
    embeddings = embeddings[keep_mask]
    paths = [paths[i] for i in range(len(paths)) if keep_mask[i]]
    orig_indices = [orig_indices[i] for i in range(len(orig_indices)) if keep_mask[i]]
    labels = labels[keep_mask]
    noise_removed_total += noise_removed

    if len(embeddings) == 0:
        return embeddings, paths, orig_indices, labels, noise_removed_total, outlier_removed_total

    new_labels = labels.copy()
    outlier_mask = np.zeros(len(embeddings), dtype=bool)
    for lbl in np.unique(labels):
        cluster_mask = labels == lbl
        cluster_embs = embeddings[cluster_mask]
        centroid = cluster_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        dists = 1 - (cluster_embs @ centroid)
        threshold = np.percentile(dists, outlier_percentile)
        outlier_mask_cluster = dists > threshold
        outlier_mask[cluster_mask] = outlier_mask_cluster

    outlier_removed = outlier_mask.sum()
    outlier_removed_total += outlier_removed

    keep_mask = ~outlier_mask
    embeddings = embeddings[keep_mask]
    paths = [paths[i] for i in range(len(paths)) if keep_mask[i]]
    orig_indices = [orig_indices[i] for i in range(len(orig_indices)) if keep_mask[i]]
    labels = labels[keep_mask]
    new_labels = new_labels[keep_mask]

    valid_labels = []
    for lbl in np.unique(new_labels):
        if np.sum(new_labels == lbl) >= min_cluster_size:
            valid_labels.append(lbl)
    if valid_labels:
        cluster_keep_mask = np.isin(new_labels, valid_labels)
        removed_after = (~cluster_keep_mask).sum()
        noise_removed_total += removed_after
        embeddings = embeddings[cluster_keep_mask]
        paths = [paths[i] for i in range(len(paths)) if cluster_keep_mask[i]]
        orig_indices = [orig_indices[i] for i in range(len(orig_indices)) if cluster_keep_mask[i]]
        labels = new_labels[cluster_keep_mask]
    else:
        noise_removed_total += len(embeddings)
        embeddings = np.empty((0, embeddings.shape[1]), dtype=embeddings.dtype)
        paths = []
        orig_indices = []
        labels = np.array([], dtype=int)

    return embeddings, paths, orig_indices, labels, noise_removed_total, outlier_removed_total


# ---------------------------------------------------------------------------
# Пропорциональный отбор
# ---------------------------------------------------------------------------
def proportional_selection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    paths: list[Path],
    orig_indices: list[int],
    num_images: int,
    seed: int,
    cluster_sizes: dict[int, int],
    global_centroid: np.ndarray,
) -> list[int]:
    """Возвращает индексы (в текущем массиве) отобранных изображений."""
    rng = np.random.default_rng(seed)
    n = embeddings.shape[0]
    if n == 0:
        raise ValueError("После очистки не осталось изображений.")

    cluster_to_indices = {}
    for idx, lbl in enumerate(labels):
        cluster_to_indices.setdefault(int(lbl), []).append(idx)

    C = len(cluster_to_indices)
    if num_images < C:
        sorted_clusters = sorted(
            cluster_to_indices.items(),
            key=lambda item: (-cluster_sizes.get(item[0], 0), item[0]),
        )
        chosen_clusters = sorted_clusters[:num_images]
        selected = []
        for _, idxs in chosen_clusters:
            pick = rng.choice(idxs, size=1, replace=False)[0]
            selected.append(pick)
        return selected

    slots = {lbl: 1 for lbl in cluster_to_indices}
    R = num_images - C
    total_valid = sum(cluster_sizes.values())
    shares = {lbl: R * (cluster_sizes[lbl] / total_valid) for lbl in cluster_to_indices}
    adds = {lbl: int(np.floor(s)) for lbl, s in shares.items()}
    rems = {lbl: s - adds[lbl] for lbl, s in shares.items()}

    remaining_slots = R - sum(adds.values())
    current_alloc = {lbl: 1 + adds[lbl] for lbl in cluster_to_indices}

    def cluster_centroid(lbl):
        idxs = cluster_to_indices[lbl]
        c = embeddings[idxs].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        return c

    centr_dist = {}
    for lbl in cluster_to_indices:
        c = cluster_centroid(lbl)
        d = 1 - np.dot(c, global_centroid)
        centr_dist[lbl] = d

    bonus_order = sorted(
        cluster_to_indices.keys(),
        key=lambda lbl: (
            rems[lbl],
            -current_alloc[lbl],
            -centr_dist[lbl],
            -lbl,
        ),
        reverse=True,
    )
    for lbl in bonus_order:
        if remaining_slots <= 0:
            break
        slots[lbl] += 1
        remaining_slots -= 1

    selected = []
    for lbl, quota in slots.items():
        idxs = cluster_to_indices[lbl]
        pick = rng.choice(idxs, size=quota, replace=False)
        selected.extend(pick.tolist())
    return selected


# ---------------------------------------------------------------------------
# TSP-сортировка (жадная)
# ---------------------------------------------------------------------------
def tsp_greedy_order(embeddings: np.ndarray, orig_indices: list[int]) -> list[int]:
    """Возвращает перестановку индексов в порядке TSP-пути."""
    N = embeddings.shape[0]
    if N == 0:
        return []
    centroid = embeddings.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    dists_to_center = 1 - (embeddings @ centroid)
    start_candidates = np.flatnonzero(dists_to_center == dists_to_center.max())
    start_idx = start_candidates[np.argmin([orig_indices[i] for i in start_candidates])]
    order = [start_idx]
    unused = set(range(N))
    unused.remove(start_idx)
    current_vec = embeddings[start_idx]
    for _ in range(1, N):
        unused_list = list(unused)
        sims = embeddings[unused_list] @ current_vec
        dists = 1 - sims
        min_dist = dists.min()
        candidates = np.where(dists == min_dist)[0]
        best_cand = min(candidates, key=lambda i: orig_indices[unused_list[i]])
        best_global = unused_list[best_cand]
        order.append(best_global)
        unused.remove(best_global)
        current_vec = embeddings[best_global]
    return order


# ---------------------------------------------------------------------------
# Генерация отчёта benchmark.txt
# ---------------------------------------------------------------------------
def compute_gini(sizes: np.ndarray) -> float:
    """Вычисление коэффициента Джини."""
    if len(sizes) == 0:
        return 0.0
    sorted_sizes = np.sort(sizes)
    n = len(sorted_sizes)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_sizes) - (n + 1) * np.sum(sorted_sizes)) / (n * np.sum(sorted_sizes))


def generate_benchmark_report(
    output_dir: Path,
    args,
    pipeline_stats: dict,
    cluster_distribution: list[dict],
    final_embeddings: np.ndarray,
    selected_indices: list[int],
    tsp_order: list[int],
    selected_paths: list[Path],
    selected_labels: np.ndarray,
    cluster_sizes: dict[int, int],
) -> str:
    """Формирует содержимое benchmark.txt согласно техническому заданию."""
    N = len(tsp_order)
    tsp_files = [selected_paths[i].name for i in tsp_order]
    tsp_clusters = [int(selected_labels[i]) for i in tsp_order] if selected_labels is not None else [-1]*N

    emb_tsp = final_embeddings[tsp_order]  # (N, D)
    dist_matrix = 1 - (emb_tsp @ emb_tsp.T)
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0.0, 2.0)

    # Сохраняем .npy файлы
    np.save(output_dir / "embeddings.npy", emb_tsp.astype(np.float32))
    np.save(output_dir / "distance_matrix.npy", dist_matrix.astype(np.float32))

    def sha256_of_file(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    emb_hash = sha256_of_file(output_dir / "embeddings.npy")
    dist_hash = sha256_of_file(output_dir / "distance_matrix.npy")

    lines = []
    sep = "=" * 60
    lines.append(sep)
    lines.append("LORA DATASET BENCHMARK REPORT")
    lines.append(sep)
    lines.append(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Utility Version: {UTILITY_VERSION}")
    lines.append("")

    # Секция 1: Configuration
    lines.append("[1. CONFIGURATION]")
    config_fields = [
        "input_dir",
        "num_images",
        "min_size",
        "max_clusters",
        "seed",
        "batch_size",
        "force",
        "min_cluster_size",
        "dedup_threshold",
        "outlier_percentile",
    ]
    for field in config_fields:
        value = getattr(args, field, None)
        lines.append(f"{field:<22s}: {value}")
    lines.append("")

    # Секция 2: Pipeline Statistics
    stats = pipeline_stats
    lines.append("[2. PIPELINE STATISTICS]")
    stat_fields = [
        ("Total files found", stats["total_found"]),
        ("Broken files", stats["broken"]),
        ("Size filtered out", stats["size_filtered"]),
        ("Valid after filter", stats["valid_after_filter"]),
        ("Duplicates removed", stats["duplicates_removed"]),
        ("Valid after dedup", stats["valid_after_dedup"]),
        ("Noise removed", stats["noise_removed"]),
        ("Outliers removed", stats["outliers_removed"]),
        ("Final pool size", stats["final_pool_size"]),
        ("Target images (N)", args.num_images),
        ("Clusters found (C)", stats["clusters_found"]),
    ]
    for name, val in stat_fields:
        lines.append(f"{name:<24s}: {val}")
    lines.append("")

    # Секция 3: Cluster Distribution
    lines.append("[3. CLUSTER DISTRIBUTION]")
    header = f"{'Cluster ID':<12s}| {'Size':<9s}| {'Selected':<10s}| {'Coverage %':<10s}"
    lines.append(header)
    lines.append("-" * 50)
    for entry in cluster_distribution:
        lines.append(
            f"{entry['cluster_id']:<12d}| {entry['size']:<9d}| {entry['selected']:<10d}| {entry['coverage']:<10.2f}"
        )
    lines.append("")

    # Секция 4: Vector Space Metrics
    triu_indices = np.triu_indices(N, k=1)
    distances = dist_matrix[triu_indices]
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    std_dist = np.std(distances)

    lines.append("[4. VECTOR SPACE METRICS (Distance Matrix)]")
    lines.append(f"Mean pairwise distance : {mean_dist:.6f}")
    lines.append(f"Median pairwise dist.  : {median_dist:.6f}")
    lines.append(f"Min pairwise distance  : {min_dist:.6f}")
    lines.append(f"Max pairwise distance  : {max_dist:.6f}")
    lines.append(f"Std deviation          : {std_dist:.6f}")
    lines.append("")

    # Секция 5: TSP PATH
    lines.append("[5. TSP PATH (FINAL ORDER)]")
    lines.append(f"{'Index':<7s}| {'File (basename)':<45s}| {'Cluster':<10s}")
    lines.append("-" * 70)
    for idx, (fname, cluster) in enumerate(zip(tsp_files, tsp_clusters)):
        lines.append(f"{idx:<7d}| {fname:<45s}| {cluster:<10d}")
    lines.append("")

    # Секция 6: File Integrity
    lines.append("[6. FILE INTEGRITY (SHA-256)]")
    lines.append(f"embeddings.npy      : {emb_hash}")
    lines.append(f"distance_matrix.npy : {dist_hash}")
    lines.append("")

    # VISIONEMBED BENCHMARK
    lines.append(sep)
    lines.append("VISIONEMBED BENCHMARK")
    lines.append(sep)
    lines.append(f"Dataset:   {args.input_dir}")
    lines.append(f"Images:    {N}")
    lines.append(f"Embedding: {emb_tsp.shape[1]}-d CLIP ViT-L/14")
    lines.append(f"Seed:      {args.seed}")
    lines.append("")

    # METRICS
    nn_dists = []
    for i in range(N):
        row = dist_matrix[i].copy()
        row[i] = np.inf
        nn_dists.append(np.min(row))
    mean_nn = np.mean(nn_dists)

    final_cluster_sizes = np.array([np.sum(selected_labels == lbl) for lbl in sorted(np.unique(selected_labels))])
    gini = compute_gini(final_cluster_sizes)

    lines.append("METRICS (cosine distance, 1 - similarity)")
    lines.append("-" * 40)
    lines.append(f"Mean pairwise distance:     {mean_dist:.4f}")
    lines.append(f"Std pairwise distance:      {std_dist:.4f}")
    lines.append(f"Min distance:               {min_dist:.4f}")
    lines.append(f"Max distance:               {max_dist:.4f}")
    lines.append(f"Mean nearest-neighbor dist: {mean_nn:.4f}  (diversity)")
    lines.append(f"Cluster balance (Gini):     {gini:.4f}  (0=perfect, 1=worst)")
    lines.append("")

    # CLUSTER COMPOSITION
    lines.append("CLUSTER COMPOSITION")
    lines.append("-" * 40)
    for lbl in sorted(np.unique(selected_labels)):
        count = np.sum(selected_labels == lbl)
        lines.append(f"  Cluster {lbl}: {count} images")
    lines.append("")

    # DISTANCE MATRIX
    lines.append("DISTANCE MATRIX")
    lines.append("-" * 40)
    max_name_len = max(len(f) for f in tsp_files) if tsp_files else 15
    header_line = " " * (max_name_len + 2) + "".join(f"{name:<{max_name_len+3}s}" for name in tsp_files)
    lines.append(header_line)
    for i, name in enumerate(tsp_files):
        row_entries = [f"{dist_matrix[i, j]:.{4}f}" for j in range(N)]
        row_line = f"{name:<{max_name_len+2}s}" + "".join(f"{val:>{max_name_len+3}s}" for val in row_entries)
        lines.append(row_line)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Подготовка датасета LoRA (FLUX) и создание benchmark-отчёта")
    parser.add_argument("--input_dir", required=True, help="Путь к папке с исходниками")
    parser.add_argument("--num_images", type=int, required=True, help="Целевой размер датасета")
    parser.add_argument("--min_size", type=int, default=512, help="Минимальный размер меньшей стороны (px)")
    parser.add_argument("--max_clusters", type=int, default=10, help="Верхняя граница для автоподбора k")
    parser.add_argument("--seed", type=int, default=43, help="Глобальный seed (по умолчанию 43)")
    parser.add_argument("--force", action="store_true", help="Перезапись папки selected/")
    parser.add_argument("--min_cluster_size", type=int, default=3, help="Защита от шума")
    parser.add_argument("--dedup_threshold", type=float, default=0.99, help="Порог косинусного сходства для дедупликации")
    parser.add_argument("--outlier_percentile", type=float, default=95.0, help="Процентиль для отсечения выбросов")
    parser.add_argument("--batch_size", type=int, default=32, help="Размер батча для инференса CLIP")
    args = parser.parse_args()

    set_seed(args.seed)

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        logger.error(f"❌ Папка {input_dir} не существует.")
        sys.exit(1)

    output_dir = input_dir / "selected"
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Папка selected/ уже существует. Используйте --force для перезаписи."
            )
        else:
            shutil.rmtree(output_dir)

    # Шаг 1. Сканирование и базовая фильтрация
    all_paths = find_images(str(input_dir))
    logger.info(f"[INFO] Сканирование: {len(all_paths)} файлов найдено")
    if not all_paths:
        logger.error("❌ Не найдено ни одного изображения.")
        sys.exit(1)

    valid_paths, original_indices, broken = load_and_filter_images(all_paths, args.min_size)
    total_found = len(all_paths)
    size_filtered = total_found - len(valid_paths) - broken
    valid_after_filter = len(valid_paths)
    logger.info(
        f"[INFO] Фильтр (>={args.min_size}px): осталось {valid_after_filter} изображений"
        + (f", {broken} битых" if broken else "")
    )
    if valid_after_filter < args.num_images:
        raise ValueError(
            f"Недостаточно валидных изображений ({valid_after_filter}) для целевого размера {args.num_images}."
        )

    # Шаг 2. Векторизация
    model_id = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    logger.info("[INFO] Загрузка модели CLIP...")
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float32)
    processor = CLIPProcessor.from_pretrained(model_id)
    embeddings = compute_embeddings(model, processor, valid_paths, args.batch_size)
    logger.info(f"[INFO] Векторизация (CPU): {valid_after_filter}/{valid_after_filter}")

    # Шаг 3. Дедупликация
    embeddings, valid_paths, original_indices, dup_removed = deduplicate_chunked(
        embeddings, valid_paths, original_indices, args.dedup_threshold
    )
    logger.info(f"[INFO] 🛡️ Дедупликация: удалено {dup_removed} копий")

    valid_after_dedup = len(valid_paths)
    if valid_after_dedup < args.num_images:
        raise ValueError(
            f"После дедупликации осталось всего {valid_after_dedup} изображений, "
            f"что меньше целевого {args.num_images}."
        )

    # Шаг 4. Кластеризация
    if valid_after_dedup < 4:
        logger.info(
            f"[INFO] Слишком мало уникальных изображений ({valid_after_dedup}), пропускаем кластеризацию и чистку."
        )
        labels = np.zeros(valid_after_dedup, dtype=int)
        k = 1
        silhouette = None
        noise_removed = 0
        outlier_removed = 0
        final_embeddings = embeddings
        final_paths = valid_paths
        final_orig_indices = original_indices
        final_labels = labels
        cluster_sizes = {0: valid_after_dedup}
    else:
        labels, k, silhouette = optimal_kmeans(embeddings, args.max_clusters, args.seed)
        logger.info(f"[INFO] Кластеризация: k={k}" + (f" (silhouette: {silhouette:.2f})" if silhouette is not None else ""))

        (
            final_embeddings,
            final_paths,
            final_orig_indices,
            final_labels,
            noise_removed,
            outlier_removed,
        ) = filter_noise_and_outliers(
            embeddings,
            labels,
            valid_paths,
            original_indices,
            args.min_cluster_size,
            args.outlier_percentile,
        )
        logger.info(f"[INFO] 🛡️ Шум: удалено {noise_removed} (микро-кластеры и схлопнувшиеся)")
        logger.info(f"[INFO] 🛡️ Выбросы: удалено {outlier_removed} (периферия > {args.outlier_percentile}%)")

        cluster_sizes = {}
        for lbl in np.unique(final_labels):
            cluster_sizes[int(lbl)] = int(np.sum(final_labels == lbl))

    final_pool_size = len(final_paths)
    if final_pool_size < args.num_images:
        raise ValueError(
            f"После всех фильтраций осталось {final_pool_size} изображений, "
            f"недостаточно для целевого размера {args.num_images}."
        )

    # Шаг 7. Пропорциональный отбор
    global_centroid = final_embeddings.mean(axis=0)
    global_centroid = global_centroid / (np.linalg.norm(global_centroid) + 1e-8)

    selected_indices = proportional_selection(
        final_embeddings,
        final_labels,
        final_paths,
        final_orig_indices,
        args.num_images,
        args.seed,
        cluster_sizes,
        global_centroid,
    )
    selected_emb = final_embeddings[selected_indices]
    selected_paths = [final_paths[i] for i in selected_indices]
    selected_orig_idx = [final_orig_indices[i] for i in selected_indices]
    selected_labels = final_labels[selected_indices] if len(final_labels) > 0 else None

    # Шаг 9. TSP-сортировка
    tsp_order = tsp_greedy_order(selected_emb, selected_orig_idx)
    logger.info("[INFO] TSP-сортировка завершена.")

    # Создаём выходную папку
    output_dir.mkdir(parents=True, exist_ok=True)

    # Копирование выбранных изображений в selected/ (с исходными именами)
    copied_count = 0
    for idx_in_selection in tsp_order:
        src = selected_paths[idx_in_selection]
        dst = output_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied_count += 1
    logger.info(f"[INFO] Скопировано {copied_count} изображений в {output_dir}")

    # Подготовка данных для отчёта
    cluster_distribution = []
    for lbl, size in cluster_sizes.items():
        selected_count = np.sum(selected_labels == lbl) if selected_labels is not None else 0
        coverage = (selected_count / size * 100) if size > 0 else 0.0
        cluster_distribution.append({
            "cluster_id": lbl,
            "size": size,
            "selected": selected_count,
            "coverage": coverage,
        })
    cluster_distribution.sort(key=lambda x: x["cluster_id"])

    pipeline_stats = {
        "total_found": total_found,
        "broken": broken,
        "size_filtered": size_filtered,
        "valid_after_filter": valid_after_filter,
        "duplicates_removed": dup_removed,
        "valid_after_dedup": valid_after_dedup,
        "noise_removed": noise_removed,
        "outliers_removed": outlier_removed,
        "final_pool_size": final_pool_size,
        "clusters_found": len(cluster_sizes) if cluster_sizes else 1,
    }

    # Генерация отчёта (включает сохранение .npy)
    report = generate_benchmark_report(
        output_dir,
        args,
        pipeline_stats,
        cluster_distribution,
        selected_emb,
        selected_indices,
        tsp_order,
        selected_paths,
        selected_labels,
        cluster_sizes,
    )

    # Запись benchmark.txt
    with open(output_dir / "benchmark.txt", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"[INFO] ✅ Готово! Датасет и отчёт сохранены в {output_dir}")
    logger.info(f"[INFO] Итоговый пул: {len(selected_paths)} изображений.")
    logger.info(f"[INFO] Файл benchmark.txt создан.")


if __name__ == "__main__":
    main()