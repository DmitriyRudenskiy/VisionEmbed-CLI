
#!/usr/bin/env python3
"""
select_lora_dataset.py  (dataset_prep)

Консольная утилита для векторного отбора изображений в датасет LoRA (FLUX).
Сканирует ТОЛЬКО текущую директорию (--input_dir), без рекурсии.

Валидация имён: только оригинальные basename, запрещены синтетические префиксы.

Артефакты в selected/:
  • <basename>.ext   — отобранные изображения (оригинальные имена)
  • embeddings.npy   — L2-нормализованные CLIP-эмбеддинги (N, D)
  • distance_matrix.npy — попарная матрица косинусных расстояний (N, N)
  • embeddings_meta.json — метаданные
  • benchmark.txt    — текстовый отчёт (7 секций)

Использование:
    python select_lora_dataset.py \\
        --input_dir ./raw_dataset \\
        --num_images 30 \\
        --min_size 512 \\
        --max_clusters 8 \\
        --seed 43 \\
        --force
"""

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Отбор изображений по векторной кластеризации для LoRA FLUX"
    )
    parser.add_argument(
        "--input_dir", required=True, type=str,
        help="Путь к директории с исходными изображениями (только текущий уровень)"
    )
    parser.add_argument(
        "--num_images", required=True, type=int,
        help="Сколько изображений отобрать"
    )
    parser.add_argument(
        "--min_size", type=int, default=512,
        help="Минимальный размер меньшей стороны (по умолчанию 512)"
    )
    parser.add_argument(
        "--max_clusters", type=int, default=10,
        help="Максимальное число кластеров (по умолчанию 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=43,
        help="Seed для воспроизводимости (по умолчанию 43)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Перезаписать существующую папку selected/"
    )
    parser.add_argument(
        "--dedup_threshold", type=float, default=0.98,
        help="Порог косинусного сходства для дедупликации (по умолчанию 0.98). "
             "0 = отключить дедупликацию."
    )
    parser.add_argument(
        "--dedup_max_remove_pct", type=float, default=80.0,
        help="Максимально допустимый процент удалённых дублей (по умолчанию 80)."
    )
    parser.add_argument(
        "--min_cluster_size", type=int, default=3,
        help="Минимальный размер кластера; кластеры меньше — шум (по умолчанию 3)"
    )
    parser.add_argument(
        "--outlier_std", type=float, default=0.0,
        help="Если > 0, отсекает выбросы внутри кластера (по умолчанию 0 = выключено)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Размер батча для CLIP (по умолчанию 32)"
    )
    return parser.parse_args()


# ─── Валидация имён ──────────────────────────────────────────────────────────

SYNTHETIC_RE = re.compile(r"^(item_|file_|img_|sample_)", re.IGNORECASE)
VALID_NAME_RE = re.compile(r"^[^\\/]+$")


def validate_filename(name: str) -> None:
    """Проверяет имя файла на соответствие naming convention."""
    if not VALID_NAME_RE.match(name):
        print(f"[ERROR] Имя файла содержит слеши: {name}", file=sys.stderr)
        sys.exit(1)
    if SYNTHETIC_RE.match(name):
        print(
            f"[ERROR] Запрещённое синтетическое имя: {name}. "
            f"Используйте оригинальные имена (MD5-хеши, UUID и т.д.)",
            file=sys.stderr,
        )
        sys.exit(1)


def scan_images(input_dir: Path) -> List[Path]:
    """Сканирует ТОЛЬКО текущую директорию (без рекурсии)."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    selected_dir = input_dir / "selected"
    files = []
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        try:
            p.relative_to(selected_dir)
            continue
        except ValueError:
            pass
        validate_filename(p.name)
        files.append(p)
    return sorted(files)


def filter_images(files: List[Path], min_size: int) -> Tuple[List[Path], int]:
    """Возвращает (valid_files, broken_count)."""
    valid = []
    broken = 0
    for p in files:
        try:
            with Image.open(p) as img:
                w, h = img.size
                if min(w, h) >= min_size:
                    valid.append(p)
        except Exception as e:
            broken += 1
            print(f"[WARN] Пропуск битого файла {p.name}: {e}", file=sys.stderr)
    return valid, broken


def get_embeddings(
    image_paths: List[Path],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, List[Path]]:
    all_embeddings: List[np.ndarray] = []
    valid_paths: List[Path] = []

    pbar = tqdm(total=len(image_paths), desc="Векторизация", unit="img")
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images: List[Image.Image] = []
        batch_valid: List[Path] = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                batch_valid.append(p)
            except Exception as e:
                print(f"[WARN] Ошибка загрузки {p.name}: {e}", file=sys.stderr)

        if not images:
            pbar.update(len(batch_paths))
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs.pooler_output
            outputs = model.visual_projection(image_embeds)
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)

        all_embeddings.append(outputs.cpu().numpy())
        valid_paths.extend(batch_valid)
        pbar.update(len(batch_paths))

    pbar.close()

    if not all_embeddings:
        return np.zeros((0, 0), dtype=np.float32), []

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    return embeddings, valid_paths


def deduplicate_embeddings(
    embeddings: np.ndarray,
    paths: List[Path],
    threshold: float,
    seed: int,
) -> Tuple[np.ndarray, List[Path], int, List[Dict[str, Any]]]:
    sim = embeddings @ embeddings.T
    n = len(paths)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        row = sim[i, i + 1 :]
        hits = np.where(row >= threshold)[0] + i + 1
        for j in hits:
            union(i, j)

    groups: dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    rng = np.random.RandomState(seed)
    keep_indices: List[int] = []
    removed = 0
    dup_groups: List[Dict[str, Any]] = []

    for group in groups.values():
        if len(group) > 1:
            removed += len(group) - 1
            chosen = int(rng.choice(group))
            keep_indices.append(chosen)
            dup_groups.append({
                "kept": paths[chosen].name,
                "removed": [paths[i].name for i in group if i != chosen],
            })
        else:
            keep_indices.append(group[0])

    keep_indices = sorted(keep_indices)
    clean_embeddings = embeddings[keep_indices]
    clean_paths = [paths[i] for i in keep_indices]
    return clean_embeddings, clean_paths, removed, dup_groups


def clusterize(
    embeddings: np.ndarray, max_clusters: int, seed: int
) -> Tuple[np.ndarray, int]:
    N = embeddings.shape[0]
    max_k = min(max_clusters, N // 2)

    if max_k < 2:
        return np.zeros(N, dtype=int), 1

    best_score = -1.0
    best_labels = np.zeros(N, dtype=int)
    best_k = 2

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    return best_labels, best_k


def filter_outliers(
    embeddings: np.ndarray, labels: np.ndarray, paths: List[Path], outlier_std: float
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if outlier_std <= 0:
        return np.ones(len(embeddings), dtype=bool), []

    mask = np.ones(len(embeddings), dtype=bool)
    outlier_info: List[Dict[str, Any]] = []

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) <= 2:
            continue

        cluster_emb = embeddings[idx]
        centroid = cluster_emb.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            continue
        centroid = centroid / norm

        dists = 1.0 - (cluster_emb @ centroid)
        mean_dist = float(dists.mean())
        std_dist = float(dists.std())
        if std_dist == 0:
            continue

        threshold = mean_dist + outlier_std * std_dist
        bad_local = np.where(dists > threshold)[0]
        bad = idx[bad_local]
        mask[bad] = False

        for bl, b in zip(bad_local, bad):
            outlier_info.append({
                "cluster_id": int(c),
                "file": paths[b].name,
                "distance": float(dists[bl]),
            })

    return mask, outlier_info


def select_images(
    image_paths: List[Path],
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_images: int,
    seed: int,
    min_cluster_size: int,
) -> Tuple[List[Path], List[int], List[int], List[Dict[str, Any]]]:
    N = len(image_paths)
    unique_labels = sorted(set(labels))
    cluster_sizes = {c: int((labels == c).sum()) for c in unique_labels}

    valid_clusters = [c for c in unique_labels if cluster_sizes[c] >= min_cluster_size]
    noise_clusters = [c for c in unique_labels if cluster_sizes[c] < min_cluster_size]

    noise_info: List[Dict[str, Any]] = []
    if noise_clusters:
        for c in noise_clusters:
            idx = np.where(labels == c)[0]
            noise_info.append({
                "cluster_id": int(c),
                "size": len(idx),
                "files": [image_paths[i].name for i in idx],
            })

    if not valid_clusters:
        print("[ERROR] Нет валидных кластеров после фильтрации шума", file=sys.stderr)
        sys.exit(1)

    valid_mask = np.array([c in valid_clusters for c in labels], dtype=bool)
    valid_paths = [image_paths[i] for i in range(N) if valid_mask[i]]
    valid_labels = labels[valid_mask]
    total_valid = len(valid_paths)

    if total_valid < num_images:
        print(
            f"[ERROR] После удаления шума осталось {total_valid} изображений, "
            f"но требуется {num_images}",
            file=sys.stderr,
        )
        sys.exit(1)

    C = len(valid_clusters)
    cluster_sizes_valid = {c: int((valid_labels == c).sum()) for c in valid_clusters}
    total = total_valid
    rng = np.random.RandomState(seed)

    if num_images < C:
        sorted_clusters = sorted(valid_clusters, key=lambda c: cluster_sizes_valid[c], reverse=True)
        chosen_clusters = sorted_clusters[:num_images]
        quotas = {c: 1 for c in chosen_clusters}
        for c in valid_clusters:
            if c not in quotas:
                quotas[c] = 0
    else:
        quotas = {c: 1 for c in valid_clusters}
        remaining = num_images - C

        extras = {c: 0 for c in valid_clusters}
        fractions = {}
        for c in valid_clusters:
            raw = remaining * cluster_sizes_valid[c] / total
            extras[c] = int(raw)
            fractions[c] = raw - extras[c]

        distributed = sum(extras.values())
        leftover = remaining - distributed
        if leftover > 0:
            sorted_by_frac = sorted(valid_clusters, key=lambda c: fractions[c], reverse=True)
            for c in sorted_by_frac[:leftover]:
                extras[c] += 1

        for c in valid_clusters:
            quotas[c] += extras[c]

    selected_paths: List[Path] = []
    selected_indices: List[int] = []
    selected_clusters: List[int] = []

    for c in valid_clusters:
        q = quotas[c]
        if q <= 0:
            continue
        idx_in_cluster = np.where(valid_labels == c)[0]
        pick = min(q, len(idx_in_cluster))
        chosen = rng.choice(idx_in_cluster, size=pick, replace=False)
        for idx in chosen:
            selected_paths.append(valid_paths[idx])
            selected_indices.append(int(idx))
            selected_clusters.append(c)

    if len(selected_paths) < num_images:
        used_local = set(selected_indices)
        available_local = [i for i in range(total_valid) if i not in used_local]
        need = num_images - len(selected_paths)
        extra = rng.choice(available_local, size=min(need, len(available_local)), replace=False)
        for idx in extra:
            selected_paths.append(valid_paths[idx])
            selected_indices.append(int(idx))
            selected_clusters.append(int(valid_labels[idx]))

    if len(selected_paths) > num_images:
        drop = rng.choice(len(selected_paths), size=len(selected_paths) - num_images, replace=False)
        keep_mask = np.ones(len(selected_paths), dtype=bool)
        keep_mask[drop] = False
        selected_paths = [p for i, p in enumerate(selected_paths) if keep_mask[i]]
        selected_indices = [idx for i, idx in enumerate(selected_indices) if keep_mask[i]]
        selected_clusters = [c for i, c in enumerate(selected_clusters) if keep_mask[i]]

    return selected_paths, selected_indices, selected_clusters, noise_info


def greedy_sort(embeddings: np.ndarray) -> List[int]:
    N = embeddings.shape[0]
    mean = embeddings.mean(axis=0)
    dists_to_mean = np.linalg.norm(embeddings - mean, axis=1)
    start = int(np.argmax(dists_to_mean))

    order = [start]
    visited = {start}

    for _ in range(N - 1):
        last = embeddings[order[-1]]
        sims = embeddings @ last
        dists = 1.0 - sims
        for idx in visited:
            dists[idx] = np.inf
        next_idx = int(np.argmin(dists))
        order.append(next_idx)
        visited.add(next_idx)

    return order


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def write_benchmark(
    selected_dir: Path,
    input_dir: Path,
    dist_mat: np.ndarray,
    meta: List[Dict[str, Any]],
    labels: np.ndarray,
    seed: int,
    stats: Dict[str, Any],
) -> None:
    """Создаёт benchmark.txt — полный отчёт из 7 секций."""
    N = dist_mat.shape[0]
    names = [m["file"] for m in meta]

    triu = dist_mat[np.triu_indices(N, k=1)]
    mean_dist = float(triu.mean())
    median_dist = float(np.median(triu))
    min_dist = float(triu.min())
    max_dist = float(triu.max())
    std_dist = float(triu.std())

    nn_dists = []
    for i in range(N):
        row = dist_mat[i].copy()
        row[i] = np.inf
        nn_dists.append(float(row.min()))
    mean_nn = float(np.mean(nn_dists))

    cluster_ids = [m["cluster_id"] for m in meta]
    counts = Counter(cluster_ids)
    if len(counts) > 1:
        total = sum(counts.values())
        shares = sorted([c / total for c in counts.values()])
        cumsum = np.cumsum(shares)
        gini = 1 - 2 * np.sum(cumsum[:-1] * np.diff(np.concatenate(([0.0], cumsum))))
    else:
        gini = 0.0

    # SHA-256 для .npy файлов
    emb_hash = sha256_file(selected_dir / "embeddings.npy")
    dm_hash = sha256_file(selected_dir / "distance_matrix.npy")

    lines: List[str] = []
    sep = "=" * 60

    lines.append(sep)
    lines.append("LORA DATASET BENCHMARK REPORT")
    lines.append(sep)
    lines.append(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    lines.append("Utility Version: 3.4 (Robust Tensor Extraction)")
    lines.append("")

    # [1. CONFIGURATION]
    lines.append("[1. CONFIGURATION]")
    lines.append(f"input_dir             : {input_dir}")
    lines.append(f"num_images            : {stats['num_images']}")
    lines.append(f"min_size              : {stats['min_size']}")
    lines.append(f"max_clusters          : {stats['max_clusters']}")
    lines.append(f"seed                  : {seed}")
    lines.append(f"batch_size            : {stats['batch_size']}")
    lines.append(f"force                 : {stats['force']}")
    lines.append(f"min_cluster_size      : {stats['min_cluster_size']}")
    lines.append(f"dedup_threshold       : {stats['dedup_threshold']}")
    lines.append(f"outlier_percentile    : {stats['outlier_std']}")
    lines.append("")

    # [2. PIPELINE STATISTICS]
    lines.append("[2. PIPELINE STATISTICS]")
    lines.append(f"Total files found        : {stats['total_found']}")
    lines.append(f"Broken files             : {stats['broken']}")
    lines.append(f"Size filtered out        : {stats['size_filtered']}")
    lines.append(f"Valid after filter       : {stats['valid_after_filter']}")
    lines.append(f"Duplicates removed       : {stats['duplicates_removed']}")
    lines.append(f"Valid after dedup        : {stats['valid_after_dedup']}")
    lines.append(f"Noise removed            : {stats['noise_removed']}")
    lines.append(f"Outliers removed         : {stats['outliers_removed']}")
    lines.append(f"Final pool size          : {stats['final_pool']}")
    lines.append(f"Target images (N)        : {stats['num_images']}")
    lines.append(f"Clusters found (C)       : {stats['clusters_found']}")
    lines.append("")

    # [3. CLUSTER DISTRIBUTION]
    lines.append("[3. CLUSTER DISTRIBUTION]")
    lines.append("Cluster ID   | Size     | Selected   | Coverage %")
    lines.append("-" * 50)
    for cid in sorted(counts.keys()):
        size = int((labels == cid).sum())
        sel = counts[cid]
        cov = (sel / size * 100) if size > 0 else 0.0
        lines.append(f"{cid:<12} | {size:<8} | {sel:<10} | {cov:.2f}")
    lines.append("")

    # [4. VECTOR SPACE METRICS]
    lines.append("[4. VECTOR SPACE METRICS (Distance Matrix)]")
    lines.append(f"Mean pairwise distance : {mean_dist:.6f}")
    lines.append(f"Median pairwise dist.  : {median_dist:.6f}")
    lines.append(f"Min pairwise distance  : {min_dist:.6f}")
    lines.append(f"Max pairwise distance  : {max_dist:.6f}")
    lines.append(f"Std deviation          : {std_dist:.6f}")
    lines.append("")

    # [5. TSP PATH (FINAL ORDER)]
    lines.append("[5. TSP PATH (FINAL ORDER)]")
    lines.append("Index   | File (basename)                          | Cluster  ")
    lines.append("-" * 70)
    for i, m in enumerate(meta):
        fname = m["file"]
        cid = m["cluster_id"]
        lines.append(f"{i:<7} | {fname:<40} | {cid}")
    lines.append("")

    # [6. FILE INTEGRITY]
    lines.append("[6. FILE INTEGRITY (SHA-256)]")
    lines.append(f"embeddings.npy      : {emb_hash}")
    lines.append(f"distance_matrix.npy : {dm_hash}")
    lines.append("")

    # [7. VISIONEMBED BENCHMARK]
    lines.append(sep)
    lines.append("VISIONEMBED BENCHMARK")
    lines.append(sep)
    lines.append(f"Dataset:   {input_dir}")
    lines.append(f"Images:    {N}")
    lines.append(f"Embedding: 768-d CLIP ViT-L/14")
    lines.append(f"Seed:      {seed}")
    lines.append("")
    lines.append("METRICS (cosine distance, 1 - similarity)")
    lines.append("-" * 40)
    lines.append(f"Mean pairwise distance:     {mean_dist:.4f}")
    lines.append(f"Std pairwise distance:      {std_dist:.4f}")
    lines.append(f"Min distance:               {min_dist:.4f}")
    lines.append(f"Max distance:               {max_dist:.4f}")
    lines.append(f"Mean nearest-neighbor dist: {mean_nn:.4f}  (diversity)")
    lines.append(f"Cluster balance (Gini):     {gini:.4f}  (0=perfect, 1=worst)")
    lines.append("")

    lines.append("CLUSTER COMPOSITION")
    lines.append("-" * 40)
    for cid in sorted(counts.keys()):
        lines.append(f"  Cluster {cid}: {counts[cid]} images")
    lines.append("")

    # DISTANCE MATRIX
    lines.append("DISTANCE MATRIX")
    lines.append("-" * 40)
    if N <= 50:
        max_name_len = max(len(n) for n in names)
        col_w = max(max_name_len + 2, 12)
        header = " " * col_w + "".join(f"{n:>{col_w}}" for n in names)
        lines.append(header)
        for i in range(N):
            row_str = f"{names[i]:<{col_w}}" + "".join(f"{dist_mat[i, j]:>{col_w}.4f}" for j in range(N))
            lines.append(row_str)
    else:
        lines.append(f"Matrix too large ({N}x{N}); showing top-left 10x10:")
        show = min(10, N)
        short_names = [n[:8] for n in names[:show]]
        header = "         " + " ".join(f"{n:>10}" for n in short_names)
        lines.append(header)
        for i in range(show):
            row_str = f"{short_names[i]:8} " + " ".join(f"{dist_mat[i, j]:10.4f}" for j in range(show))
            lines.append(row_str)
        lines.append("...")
    lines.append("")
    lines.append("Use this file to compare datasets via chat / LLM.")
    lines.append("Copy benchmark.txt from two runs and ask for comparison.")

    bench_path = selected_dir / "benchmark.txt"
    bench_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] benchmark.txt создан: {bench_path}")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()

    if not input_dir.exists():
        print(f"[ERROR] Директория не найдена: {input_dir}", file=sys.stderr)
        sys.exit(1)

    selected_dir = input_dir / "selected"
    if selected_dir.exists() and not args.force:
        print(
            f"[ERROR] Папка {selected_dir} уже существует. "
            f"Используйте --force для перезаписи.",
            file=sys.stderr,
        )
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[INFO] Сканирование: {input_dir} (только текущая директория)")
    all_files = scan_images(input_dir)
    total_found = len(all_files)
    print(f"[INFO] Найдено файлов: {total_found}")

    valid_files, broken = filter_images(all_files, args.min_size)
    size_filtered = total_found - len(valid_files) - broken
    valid_after_filter = len(valid_files)
    print(f"[INFO] Прошло фильтр по размеру: {valid_after_filter}")

    if len(valid_files) < args.num_images:
        print(
            f"[ERROR] Найдено {len(valid_files)} валидных изображений, "
            f"но требуется {args.num_images}",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Загрузка CLIP ({device})...")
    model_name = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings, valid_files = get_embeddings(
        valid_files, processor, model, device, batch_size=args.batch_size
    )
    print(f"[INFO] Эмбеддинги: {embeddings.shape}")

    if len(valid_files) < args.num_images:
        print(
            f"[ERROR] После загрузки осталось {len(valid_files)} изображений, "
            f"требуется {args.num_images}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Дедупликация
    dup_groups: List[Dict[str, Any]] = []
    duplicates_removed = 0
    if args.dedup_threshold > 0:
        embeddings, valid_files, duplicates_removed, dup_groups = deduplicate_embeddings(
            embeddings, valid_files, args.dedup_threshold, args.seed
        )
        total_before = len(valid_files) + duplicates_removed
        pct = (duplicates_removed / total_before) * 100 if total_before > 0 else 0
        print(
            f"[INFO] Дедупликация: удалено {duplicates_removed} дублей из {total_before} "
            f"({pct:.1f}%), осталось {len(valid_files)}"
        )
        if pct > args.dedup_max_remove_pct:
            print(
                f"[ERROR] Удалено {pct:.1f}% дублей, что превышает порог "
                f"--dedup_max_remove_pct={args.dedup_max_remove_pct}%.",
                file=sys.stderr,
            )
            sys.exit(1)
        if duplicates_removed:
            print("[INFO] Удалённые дубли:")
            for g in dup_groups:
                print(f"       Сохранён: {g['kept']}")
                for r in g["removed"]:
                    print(f"         └─ дубль: {r}")
        else:
            print(f"[INFO] Дедупликация: дублей не найдено")

    valid_after_dedup = len(valid_files)

    if len(valid_files) < args.num_images:
        print(
            f"[ERROR] После дедупликации осталось {len(valid_files)} изображений, "
            f"требуется {args.num_images}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Кластеризация
    labels, n_clusters = clusterize(embeddings, args.max_clusters, args.seed)
    print(f"[INFO] Кластеризация: определено {n_clusters} кластеров")

    # 3. Отсечение выбросов внутри кластера (опционально)
    outlier_info: List[Dict[str, Any]] = []
    outliers_removed = 0
    if args.outlier_std > 0:
        outlier_mask, outlier_info = filter_outliers(
            embeddings, labels, valid_files, args.outlier_std
        )
        outliers_removed = int((~outlier_mask).sum())
        if outliers_removed:
            embeddings = embeddings[outlier_mask]
            valid_files = [p for p, m in zip(valid_files, outlier_mask) if m]
            labels = labels[outlier_mask]
            print(
                f"[INFO] Отсечено {outliers_removed} внутрикластерных выбросов "
                f"(outlier_std={args.outlier_std})"
            )
            if outlier_info:
                print("[INFO] Отсечённые выбросы:")
                for o in outlier_info:
                    print(
                        f"       Кластер {o['cluster_id']}: "
                        f"{o['file']} (distance={o['distance']:.4f})"
                    )
            labels, n_clusters = clusterize(embeddings, args.max_clusters, args.seed)
            print(f"[INFO] Перекластеризация: {n_clusters} кластеров")

    final_pool = len(valid_files)

    if len(valid_files) < args.num_images:
        print(
            f"[ERROR] После фильтрации выбросов осталось {len(valid_files)} изображений, "
            f"требуется {args.num_images}",
            file=sys.stderr,
        )
        sys.exit(1)

    # 4. Отбор с фильтром мелких кластеров
    sel_paths, sel_indices, sel_clusters, noise_info = select_images(
        valid_files,
        embeddings,
        labels,
        args.num_images,
        args.seed,
        min_cluster_size=args.min_cluster_size,
    )
    sel_embeddings = embeddings[sel_indices]

    noise_removed = sum(n["size"] for n in noise_info)

    # Статистика
    cluster_counter = Counter(sel_clusters)
    valid_unique = sorted(set(labels))
    valid_sizes = {c: int((labels == c).sum()) for c in valid_unique}
    print("[INFO] Распределение (валидные кластеры):")
    for c in sorted(cluster_counter.keys()):
        orig_size = valid_sizes.get(c, 0)
        print(f"       - Кластер {c}: {orig_size} шт. → отобрано {cluster_counter[c]}")

    if noise_info:
        print("[INFO] Проигнорированы шумовые кластеры:")
        for n in noise_info:
            print(f"       Кластер {n['cluster_id']} ({n['size']} шт.):")
            for f in n["files"]:
                print(f"         └─ {f}")

    # 5. Жадная сортировка векторов
    order = greedy_sort(sel_embeddings)
    sel_paths = [sel_paths[i] for i in order]
    sel_indices = [sel_indices[i] for i in order]
    sel_clusters = [sel_clusters[i] for i in order]
    sel_embeddings = sel_embeddings[order]

    print(f"[INFO] Сортировка векторов: жадный путь (длина {len(order)})")

    # 6. Сохранение — оригинальные имена
    selected_dir.mkdir(parents=True, exist_ok=True)

    meta = []
    for i, (src_path, clust_id) in enumerate(zip(sel_paths, sel_clusters), start=1):
        dst_name = src_path.name
        dst_path = selected_dir / dst_name
        # Если файл уже есть (не должно, но на всякий случай)
        if dst_path.exists():
            base = src_path.stem
            ext = src_path.suffix.lower()
            dst_name = f"{base}_{i}{ext}"
            dst_path = selected_dir / dst_name
        shutil.copy2(src_path, dst_path)

        cluster_size = int((labels == clust_id).sum()) if clust_id in valid_sizes else 0
        meta.append(
            {
                "index": i - 1,
                "file": dst_name,
                "original_path": src_path.name,
                "cluster_id": int(clust_id),
                "cluster_size": cluster_size,
            }
        )

    np.save(selected_dir / "embeddings.npy", sel_embeddings.astype(np.float32))

    dist_mat = 1.0 - (sel_embeddings @ sel_embeddings.T)
    np.fill_diagonal(dist_mat, 0.0)
    np.save(selected_dir / "distance_matrix.npy", dist_mat.astype(np.float32))

    with open(selected_dir / "embeddings_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Статистика для отчёта
    stats = {
        "num_images": args.num_images,
        "min_size": args.min_size,
        "max_clusters": args.max_clusters,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "force": args.force,
        "min_cluster_size": args.min_cluster_size,
        "dedup_threshold": args.dedup_threshold,
        "outlier_std": args.outlier_std,
        "total_found": total_found,
        "broken": broken,
        "size_filtered": size_filtered,
        "valid_after_filter": valid_after_filter,
        "duplicates_removed": duplicates_removed,
        "valid_after_dedup": valid_after_dedup,
        "noise_removed": noise_removed,
        "outliers_removed": outliers_removed,
        "final_pool": final_pool,
        "clusters_found": n_clusters,
    }

    # benchmark.txt
    write_benchmark(selected_dir, input_dir, dist_mat, meta, labels, args.seed, stats)

    print(
        f"[INFO] Сохранение: embeddings.npy {sel_embeddings.shape}, "
        f"distance_matrix.npy {dist_mat.shape}"
    )
    print(f"[INFO] Результат: {selected_dir} ({len(sel_paths)} файлов)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()