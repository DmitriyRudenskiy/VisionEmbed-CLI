import os
import sys
import json
import random
import logging
import shutil
import re
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import transformers
import huggingface_hub
from tqdm import tqdm

# Попытка импортировать FAISS для быстрой дедупликации
try:
    import faiss

    USE_FAISS = True
except ImportError:
    USE_FAISS = False

# ==============================================================================
# 1. LOGGING & VALIDATION
# ==============================================================================

transformers.logging.set_verbosity_error()
huggingface_hub.logging.set_verbosity_error()
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

SYNTHETIC_PATTERN = re.compile(r"^(item_|file_|img_|sample_)\d+", re.IGNORECASE)
SOFT_DUP_THRESHOLD = 0.025
DS_NORMALIZATION_FACTOR = 0.35


def validate_filename(filename: str) -> bool:
    if '/' in filename or '\\' in filename:
        return False
    if SYNTHETIC_PATTERN.match(filename):
        return False
    if not re.match(r"^.+\..+$", filename):
        return False
    parts = filename.split('.')
    if len(parts) > 2 and parts[-1].lower() == parts[-2].lower():
        return False
    return True


def set_global_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_gini(sizes):
    if not sizes or sum(sizes) == 0:
        return 0.0
    n = len(sizes)
    sizes = np.array(sorted(sizes), dtype=float)
    index = np.arange(1, n + 1)
    return np.sum((2 * index - n - 1) * sizes) / (n * np.sum(sizes))


# ==============================================================================
# 2. HELPER CLASSES
# ==============================================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def load_and_filter_images(input_dir: str, min_size: int, allow_synthetic: bool):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
    all_paths = []

    entries = os.listdir(input_dir)
    for f in entries:
        full_path = os.path.abspath(os.path.join(input_dir, f))
        if os.path.isfile(full_path) and Path(f).suffix.lower() in valid_extensions:
            if not validate_filename(f):
                if not allow_synthetic:
                    logging.warning(f"File '{f}' has synthetic name. Skipping. Use --allow_synthetic_names to include.")
                    continue
            all_paths.append(full_path)

    all_paths.sort()
    valid_data, broken_count, size_filtered_count = [], 0, 0

    print(f"[INFO] Scanning (no recursion): {len(all_paths)} files found")
    for idx, path in enumerate(all_paths):
        try:
            with Image.open(path) as img:
                w, h = img.size
                if min(w, h) < min_size:
                    size_filtered_count += 1
                    continue
                img.convert("RGB").load()
            valid_data.append((idx, path, os.path.getsize(path)))
        except Exception:
            broken_count += 1

    return valid_data, len(all_paths), broken_count, size_filtered_count


# ==============================================================================
# 3. QUALITY METRICS (v5.0)
# ==============================================================================

def calculate_quality_scores(N, dist_matrix, clusters_meta, items_meta, ss_score):
    coverages = []
    selected_counts = {}
    for cid, meta in clusters_meta.items():
        sel_count = sum(1 for item in items_meta if item['cluster_id'] == cid)
        selected_counts[cid] = sel_count
        cov = (sel_count / meta['size'] * 100) if meta['size'] > 0 else 0
        coverages.append(cov)

    mean_coverage = np.mean(coverages) if coverages else 0.0

    sum_cov = sum(coverages)
    ce = 0.0
    if sum_cov > 0:
        p = np.array(coverages) / sum_cov
        p = p[p > 0]
        ce = -np.sum(p * np.log(p))

    ds, mean_pairwise, std_pairwise, mean_nn, intra_min_dist, soft_dup_pairs = 0.0, 0.0, 0.0, 0.0, 0.0, 0

    if N > 1:
        i, j = np.triu_indices(N, k=1)
        upper_dists = dist_matrix[i, j]

        mean_pairwise = np.mean(upper_dists)
        std_pairwise = np.std(upper_dists)

        temp_matrix = dist_matrix.copy()
        np.fill_diagonal(temp_matrix, np.inf)
        nn_dists = np.min(temp_matrix, axis=1)
        mean_nn = np.mean(nn_dists)

        intra_min_dist = float(np.min(upper_dists))
        soft_dup_pairs = int(np.sum(upper_dists < SOFT_DUP_THRESHOLD))

        ds = (mean_pairwise * mean_nn) / std_pairwise if std_pairwise > 0 else 0.0

    gini = calculate_gini(list(selected_counts.values()))

    norm_ds = min(ds / DS_NORMALIZATION_FACTOR, 1.0)
    norm_gini = 1.0 - gini
    norm_cov = mean_coverage / 100.0
    norm_ss = max(0.0, min(ss_score, 1.0))

    lrs = 0.3 * norm_ds + 0.25 * norm_gini + 0.25 * norm_cov + 0.2 * norm_ss

    if mean_coverage < 60.0:
        lrs *= 0.7
    elif mean_coverage < 70.0:
        lrs *= 0.85

    if intra_min_dist < SOFT_DUP_THRESHOLD and N > 1:
        lrs *= 0.9

    if lrs >= 0.80:
        grade = "GOOD"
    elif lrs >= 0.65:
        grade = "ACCEPTABLE"
    else:
        grade = "POOR"

    return {
        "ds": ds, "ce": ce, "ss": ss_score, "gini": gini,
        "mean_coverage": mean_coverage, "intra_min_dist": intra_min_dist,
        "soft_dup_pairs": soft_dup_pairs, "lrs": lrs, "grade": grade
    }


# ==============================================================================
# 4. COVER IMAGE SELECTION
# ==============================================================================

def select_cover_image(sel_embs, sel_global_indices):
    N = sel_embs.shape[0]
    if N == 0:
        return None, None

    centroid = np.mean(sel_embs, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    else:
        centroid = np.zeros_like(centroid)

    dists_to_center = 1.0 - (sel_embs @ centroid)
    min_dist = np.min(dists_to_center)

    candidates = np.where(np.abs(dists_to_center - min_dist) < 1e-6)[0]
    global_indices_array = np.array(sel_global_indices)
    cover_idx = candidates[np.argmin(global_indices_array[candidates])]

    return int(cover_idx), float(min_dist)


# ==============================================================================
# 5. BENCHMARK REPORT (text)
# ==============================================================================

def generate_benchmark(out_dir, args, stats, clusters_meta, dist_matrix, items_meta, qs, cover_info,
                       strategy_comparison, emb_dim):
    report = []
    sep = "=" * 60
    N = dist_matrix.shape[0]

    report.append(sep)
    report.append("LORA DATASET BENCHMARK REPORT")
    report.append(sep)
    report.append(f"Timestamp: {datetime.now().isoformat()}")
    report.append("")

    report.append("[1. CONFIGURATION]")
    for k, v in vars(args).items():
        report.append(f"{k:<22}: {v}")
    report.append("")

    report.append("[2. PIPELINE STATISTICS]")
    for k, v in stats.items():
        report.append(f"{k:<25}: {v}")
    report.append("")

    report.append("[3. STRATEGY GRID SEARCH (Auto-Selection)]")
    report.append(
        f"{'#':<4} | {'K offset':<10} | {'Mode':<13} | {'N':<5} | {'DS':<8} | {'Gini':<8} | {'Cov%':<8} | {'LRS':<8} | {'Grade':<12} | {'Status'}")
    report.append("-" * 115)
    for entry in strategy_comparison:
        status = "SELECTED" if entry.get('selected', False) else ""
        report.append(
            f"{entry['strategy_idx']:<4} | {entry['k_offset']:<10} | {entry['mode']:<13} | {entry['N']:<5} | "
            f"{entry['ds']:<8.4f} | {entry['gini']:<8.4f} | {entry['mean_coverage']:<8.2f} | "
            f"{entry['lrs']:<8.4f} | {entry['grade']:<12} | {status}"
        )
    report.append("")

    report.append("[4. CLUSTER DISTRIBUTION]")
    report.append(f"{'Cluster ID':<12} | {'Size':<8} | {'Selected':<10} | {'Coverage %':<10}")
    report.append("-" * 50)
    for cid, meta in sorted(clusters_meta.items()):
        sel_count = sum(1 for item in items_meta if item['cluster_id'] == cid)
        coverage = (sel_count / meta['size'] * 100) if meta['size'] > 0 else 0
        report.append(f"{cid:<12} | {meta['size']:<8} | {sel_count:<10} | {coverage:<10.2f}")
    report.append("")

    report.append("[5. QUALITY SCORES v5.0]")
    report.append(f"Diversity Score (DS)       : {qs['ds']:.4f}")
    report.append(f"Coverage Entropy (CE)      : {qs['ce']:.4f}")
    report.append(f"Silhouette Score (SS)      : {qs['ss']:.4f}")
    report.append(f"Cluster Balance (Gini)     : {qs['gini']:.4f}")
    report.append(f"Mean Coverage              : {qs['mean_coverage']:.2f}%")
    min_dist_str = f"{qs['intra_min_dist']:.4f}"
    if qs['intra_min_dist'] < SOFT_DUP_THRESHOLD and N > 1:
        min_dist_str += " WARN"
    report.append(f"Intra-cluster Min Distance : {min_dist_str}")
    dup_str = f"{qs['soft_dup_pairs']} pairs"
    if qs['soft_dup_pairs'] > 0:
        dup_str += " WARN"
    report.append(f"Soft Duplicates Found      : {dup_str}")
    report.append(f"LoRA Readiness Score (LRS) : {qs['lrs']:.4f} [{qs['grade']}]")
    report.append("")

    report.append("[6. COVER IMAGE]")
    if cover_info:
        report.append(f"File                  : {cover_info['file']}")
        report.append(f"Cluster ID            : {cover_info['cluster_id']}")
        report.append(f"Distance to centroid  : {cover_info['dist_to_center']:.6f}")
        report.append(f"TSP Index             : {cover_info['tsp_index']}")
        report.append(f"Original path         : {cover_info['original_path']}")
    else:
        report.append("(no cover selected)")
    report.append("")

    report.append(sep)
    report.append("VISIONEMBED BENCHMARK")
    report.append(sep)
    report.append(f"Dataset:   {args.input_dir}")
    report.append(f"Images:    {N}")
    report.append(f"Embedding: {emb_dim}-d {args.clip_model}")
    report.append(f"Seed:      {args.seed}")
    if cover_info:
        report.append(f"Cover:     {cover_info['file']}")
    report.append("")
    report.append("METRICS (cosine distance, 1 - similarity)")
    report.append("----------------------------------------")
    if N > 1:
        i, j = np.triu_indices(N, k=1)
        upper_dists = dist_matrix[i, j]
        report.append(f"Mean pairwise distance:     {np.mean(upper_dists):.4f}")
        report.append(f"Std pairwise distance:      {np.std(upper_dists):.4f}")
        report.append(f"Min distance:               {np.min(upper_dists):.4f}")
        report.append(f"Max distance:               {np.max(upper_dists):.4f}")
        temp_matrix = dist_matrix.copy()
        np.fill_diagonal(temp_matrix, np.inf)
        report.append(f"Mean nearest-neighbor dist: {np.mean(np.min(temp_matrix, axis=1)):.4f}  (diversity)")

    report.append(f"Cluster balance (Gini):     {qs['gini']:.4f}  (0=perfect, 1=worst)")
    report.append("")

    report.append("CLUSTER COMPOSITION")
    report.append("----------------------------------------")
    selected_counts = {}
    for item in items_meta:
        cid = item['cluster_id']
        selected_counts[cid] = selected_counts.get(cid, 0) + 1
    for cid in sorted(selected_counts.keys()):
        report.append(f"  Cluster {cid}: {selected_counts[cid]} images")
    report.append("")

    with open(os.path.join(out_dir, "benchmark.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))


# ==============================================================================
# 6. QUOTA DISTRIBUTION
# ==============================================================================

def compute_quotas(clusters_meta, N, mode):
    C = len(clusters_meta)
    if C == 0:
        return {}

    quotas = {cid: 0 for cid in clusters_meta}

    if N <= C:
        sorted_cids = sorted(clusters_meta.keys(), key=lambda c: (-clusters_meta[c]["size"], c))
        for cid in sorted_cids[:N]:
            quotas[cid] = 1
        return quotas

    for cid in clusters_meta:
        quotas[cid] = 1
    remaining = N - C

    if mode == "uniform":
        weights = {cid: 1.0 for cid in clusters_meta}
    elif mode == "sqrt":
        weights = {cid: np.sqrt(meta["size"]) for cid, meta in clusters_meta.items()}
    elif mode == "log":
        weights = {cid: np.log1p(meta["size"]) for cid, meta in clusters_meta.items()}
    elif mode == "inverse":
        weights = {cid: 1.0 / max(1, meta["size"]) for cid, meta in clusters_meta.items()}
    else:  # "proportional"
        weights = {cid: float(meta["size"]) for cid, meta in clusters_meta.items()}

    total_weight = sum(weights.values())

    adds = {}
    for cid in clusters_meta:
        raw = remaining * (weights[cid] / total_weight)
        adds[cid] = int(np.floor(raw))

    for cid in clusters_meta:
        quotas[cid] += adds[cid]

    for cid in clusters_meta:
        if quotas[cid] > clusters_meta[cid]["size"]:
            quotas[cid] = clusters_meta[cid]["size"]

    remaining_slots = N - sum(quotas.values())

    if remaining_slots > 0:
        centers = np.array([clusters_meta[cid]["center"] for cid in clusters_meta])
        global_center = np.mean(centers, axis=0)
        norm_gc = np.linalg.norm(global_center)
        if norm_gc > 0:
            global_center /= norm_gc

        dists_to_global = {}
        for cid, meta in clusters_meta.items():
            sim = np.dot(meta["center"], global_center)
            if hasattr(sim, "item"):
                sim = sim.item()
            dists_to_global[cid] = max(0.0, 1.0 - sim)

        sorted_cids = sorted(
            clusters_meta.keys(),
            key=lambda c: (
                -(clusters_meta[c]["size"] - quotas[c]),
                dists_to_global[c],
                c
            )
        )

        idx = 0
        while remaining_slots > 0 and idx < len(sorted_cids) * 2:
            cid = sorted_cids[idx % len(sorted_cids)]
            if quotas[cid] < clusters_meta[cid]["size"]:
                quotas[cid] += 1
                remaining_slots -= 1
            idx += 1

    return quotas


def build_selection_for_mode(clusters_meta, N, mode, valid_data_dedup, seed, strategy_idx, full_dist_matrix, ss_score):
    # Использование локального сида для независимости стратегий
    rng = random.Random(seed + strategy_idx)

    quotas = compute_quotas(clusters_meta, N, mode)

    selected_indices = []
    for cid, meta in clusters_meta.items():
        if quotas[cid] > 0:
            selected_indices.extend(rng.sample(meta["members"], quotas[cid]))

    sel_embs = np.vstack([valid_data_dedup[i]["emb"] for i in selected_indices])
    sel_global_indices = [valid_data_dedup[i]["global_idx"] for i in selected_indices]

    # Использование предвычисленной матрицы для ускорения Nearest Neighbor и извлечения подматрицы
    sub_dist_matrix = full_dist_matrix[np.ix_(selected_indices, selected_indices)]

    # Nearest Neighbor heuristic (вместо TSP)
    center = np.mean(sel_embs, axis=0)
    center /= np.linalg.norm(center)
    dists_to_center = 1.0 - (sel_embs @ center)
    candidates = np.where(dists_to_center == np.max(dists_to_center))[0]
    path = [candidates[np.argmin(np.array(sel_global_indices)[candidates])]]
    used = {path[0]}

    for _ in range(N - 1):
        dists = sub_dist_matrix[path[-1]].copy()
        dists[list(used)] = np.inf
        nxt = np.argmin(dists)
        path.append(nxt)
        used.add(nxt)

    ordered_selected_indices = [selected_indices[i] for i in path]

    idx_to_cid = {m: cid for cid, meta in clusters_meta.items() for m in meta["members"]}
    items_meta = []
    for i, dedup_idx in enumerate(ordered_selected_indices):
        cid = idx_to_cid.get(dedup_idx, -1)
        items_meta.append({
            "index": i,
            "cluster_id": int(cid),
            "cluster_size": int(clusters_meta[cid]["size"]) if cid != -1 else 0,
        })

    final_embs = sel_embs[path]
    E_final = np.vstack(final_embs).astype(np.float32)

    # Берем финальную матрицу из подматрицы по TSP-пути
    dist_matrix = sub_dist_matrix[np.ix_(path, path)]

    qs = calculate_quality_scores(N, dist_matrix, clusters_meta, items_meta, ss_score)

    return {
        "mode": mode,
        "N": N,
        "quotas": quotas,
        "ordered_indices": ordered_selected_indices,
        "sel_embs": sel_embs,
        "sel_global_indices": sel_global_indices,
        "path": path,
        "items_meta": items_meta,
        "dist_matrix": dist_matrix,
        "E_final": E_final,
        "ds": qs["ds"],
        "gini": qs["gini"],
        "mean_coverage": qs["mean_coverage"],
        "lrs": qs["lrs"],
        "grade": qs["grade"],
        "soft_dup_pairs": qs["soft_dup_pairs"],
        "intra_min_dist": qs["intra_min_dist"],
        "clusters_meta": clusters_meta,
        "ss_score": ss_score
    }


# ==============================================================================
# 7. КОНСОЛЬНЫЙ ВЫВОД
# ==============================================================================

def print_structured_summary(stats, clusters_meta, items_meta, qs, best_strategy, cover_info, all_strategies, args):
    sep = "=" * 70
    print("\n" + sep)
    print("STRUCTURED PIPELINE SUMMARY".center(70))
    print(sep)

    print("\n[CONFIGURATION]")
    for k, v in vars(args).items():
        print(f"  {k:<20}: {v}")

    print("\n[STATISTICS]")
    for k, v in stats.items():
        print(f"  {k:<25}: {v}")

    print("\n[BEST STRATEGY]")
    print(f"  Strategy index      : #{best_strategy['strategy_idx']}")
    print(f"  K offset            : {best_strategy.get('k_offset', 'N/A')}")
    print(f"  Target K            : {best_strategy.get('target_k', 'N/A')}")
    print(f"  Mode                : {best_strategy['mode'].upper()}")
    print(f"  N (selected images) : {best_strategy['N']}")
    print(f"  LRS                 : {best_strategy['lrs']:.4f} ({best_strategy['grade']})")
    print(f"  DS                  : {best_strategy['ds']:.4f}")
    print(f"  Gini                : {best_strategy['gini']:.4f}")
    print(f"  Mean coverage       : {best_strategy['mean_coverage']:.1f}%")
    print(f"  Soft duplicate pairs: {best_strategy['soft_dup_pairs']}")

    print("\n[QUALITY SCORES]")
    print(f"  LRS (LoRA Readiness) : {qs['lrs']:.4f} [{qs['grade']}]")
    print(f"  DS (Diversity)       : {qs['ds']:.4f}")
    print(f"  CE (Coverage Entropy): {qs['ce']:.4f}")
    print(f"  SS (Silhouette)      : {qs['ss']:.4f}")
    print(f"  Gini (Balance)       : {qs['gini']:.4f}")
    print(f"  Mean coverage        : {qs['mean_coverage']:.2f}%")
    print(f"  Intra-min distance   : {qs['intra_min_dist']:.4f}")
    print(f"  Soft duplicate pairs : {qs['soft_dup_pairs']}")

    print("\n[CLUSTERS]")
    print(f"  {'ID':<4} | {'Size':<6} | {'Selected':<8} | {'Coverage %':<10}")
    print("  " + "-" * 35)
    for cid, meta in sorted(clusters_meta.items()):
        sel_count = sum(1 for item in items_meta if item['cluster_id'] == cid)
        coverage = (sel_count / meta['size'] * 100) if meta['size'] > 0 else 0
        print(f"  {cid:<4} | {meta['size']:<6} | {sel_count:<8} | {coverage:<10.2f}")

    if cover_info:
        print("\n[COVER IMAGE]")
        print(f"  File          : {cover_info['file']}")
        print(f"  Cluster ID    : {cover_info['cluster_id']}")
        print(f"  Distance to centroid: {cover_info['dist_to_center']:.6f}")
        print(f"  NN-TSP index  : {cover_info['tsp_index']}")
        print(f"  Original path : {cover_info['original_path']}")

    print("\n[ALL STRATEGIES (top 5 by LRS)]")
    sorted_strategies = sorted(all_strategies, key=lambda x: x['lrs'], reverse=True)[:5]
    print(f"  {'#':<4} | {'K offset':<10} | {'Mode':<13} | {'N':<5} | {'LRS':<8} | {'Grade':<12}")
    print("  " + "-" * 65)
    for s in sorted_strategies:
        print(
            f"  {s['strategy_idx']:<4} | {s['k_offset']:<10} | {s['mode']:<13} | {s['N']:<5} | {s['lrs']:<8.4f} | {s['grade']:<12}")

    print(sep + "\n")


# ==============================================================================
# 8. MAIN PIPELINE
# ==============================================================================

def run_pipeline(args):
    set_global_seed(args.seed)
    valid_data, total_found, broken_count, size_filtered_count = load_and_filter_images(
        args.input_dir, args.min_size, args.allow_synthetic_names
    )

    if args.num_images is not None and len(valid_data) < args.num_images:
        raise ValueError(f"Found {len(valid_data)} valid images, but {args.num_images} requested.")

    # Определение устройства и типа данных
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[INFO] Loading CLIP model '{args.clip_model}' on {device} with dtype {dtype}...")
    model = transformers.CLIPModel.from_pretrained(
        args.clip_model,
        torch_dtype=dtype
    ).to(device)
    processor = transformers.CLIPProcessor.from_pretrained(args.clip_model)
    model.eval()

    embeddings = []
    print("[INFO] Vectorization started...")
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_data), args.batch_size), desc="Vectorizing"):
            batch_data = valid_data[i:i + args.batch_size]
            images = [Image.open(d[1]).convert("RGB") for d in batch_data]
            inputs = processor(images=images, return_tensors="pt").to(device)

            # Явный вызов модели зрения и проекционного слоя для 100% совместимости
            vision_outputs = model.vision_model(pixel_values=inputs['pixel_values'])
            pooled_output = vision_outputs[1]  # Индекс 1 соответствует pooler_output
            image_features = model.visual_projection(pooled_output)

            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(image_features.cpu().numpy())

    print("[INFO] Vectorization completed.")
    E = np.vstack(embeddings).astype(np.float32)

    # Дедупликация (FAISS или O(N²) фоллбэк)
    N = E.shape[0]
    uf = UnionFind(N)

    if USE_FAISS:
        print("[INFO] Deduplicating with FAISS (fast)...")
        index = faiss.IndexFlatIP(E.shape[1])
        faiss.normalize_L2(E)
        index.add(E)
        lims, D, I = index.range_search(E, args.dedup_threshold)
        for i in tqdm(range(N), desc="FAISS UnionFind"):
            for j_idx in range(lims[i], lims[i + 1]):
                j = I[j_idx]
                if i != j:
                    uf.union(i, j)
    else:
        print("[INFO] Deduplicating with O(N²) blocks (FAISS not found)...")
        for i in tqdm(range(0, N, 1024), desc="Dedup Blocks"):
            for j in range(i, N, 1024):
                sim = E[i:min(i + 1024, N)] @ E[j:min(j + 1024, N)].T
                if i == j:
                    np.fill_diagonal(sim, -1.0)
                rows, cols = np.where(sim >= args.dedup_threshold)
                for r, c in zip(rows, cols):
                    uf.union(i + r, j + c)

    groups = {}
    for i in range(N):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    dedup_indices = [max(m, key=lambda x: (valid_data[x][2], -valid_data[x][0])) for m in groups.values()]
    dedup_indices.sort()
    E_dedup = E[dedup_indices]
    valid_data_dedup_raw = [valid_data[i] for i in dedup_indices]
    duplicates_removed = N - len(E_dedup)

    # Предвычисление полной матрицы расстояний для E_dedup (1 раз)
    N_valid = len(E_dedup)
    print(f"[INFO] Precomputing distance matrix for {N_valid} deduplicated items...")
    full_dist_matrix = np.zeros((N_valid, N_valid), dtype=np.float32)
    for i in tqdm(range(0, N_valid, 512), desc="Distance Matrix"):
        chunk = E_dedup[i:min(i + 512, N_valid)]
        full_dist_matrix[i:min(i + 512, N_valid)] = np.clip(1.0 - (chunk @ E_dedup.T), 0.0, 2.0)
    np.fill_diagonal(full_dist_matrix, 0.0)

    final_pool_indices, clusters_meta, noise_removed, outliers_removed, best_score = [], {}, 0, 0, 0.0

    if N_valid < 4:
        final_pool_indices = list(range(N_valid))
        clusters_meta[0] = {"size": N_valid, "members": final_pool_indices, "center": np.mean(E_dedup, axis=0)}
    else:
        max_k = min(args.max_clusters, N_valid // 2)
        if max_k < 2:
            max_k = 2
        best_k = 2
        print("[INFO] Searching for best K (Silhouette)...")
        for k in tqdm(range(2, max_k + 1), desc="KMeans Grid"):
            labels = KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit_predict(E_dedup)
            score = silhouette_score(E_dedup, labels)
            if score > best_score:
                best_score, best_k = score, k

        labels = KMeans(n_clusters=best_k, random_state=args.seed, n_init=10).fit_predict(E_dedup)
        clusters = {i: [] for i in range(best_k)}
        for idx, lbl in enumerate(labels):
            clusters[lbl].append(idx)

        valid_clusters = {cid: m for cid, m in clusters.items() if len(m) >= args.min_cluster_size}
        noise_removed = sum(len(m) for cid, m in clusters.items() if len(m) < args.min_cluster_size)

        for cid, members in valid_clusters.items():
            c_embs = E_dedup[members]
            center = np.mean(c_embs, axis=0)
            center /= np.linalg.norm(center)
            dists = 1.0 - (c_embs @ center)
            clean_members = [m for m, d in zip(members, dists) if d <= np.percentile(dists, args.outlier_percentile)]
            outliers_removed += len(members) - len(clean_members)
            if len(clean_members) >= args.min_cluster_size:
                clusters_meta[cid] = {
                    "size": len(clean_members),
                    "members": clean_members,
                    "center": center
                }
                final_pool_indices.extend(clean_members)

    if len(final_pool_indices) == 0:
        raise ValueError("Pool is empty after all filters. Cannot form dataset.")

    # AUTO N
    if args.num_images is None:
        print("[INFO] Auto N selection (adaptive coverage):")
        auto_n = 0
        for cid in sorted(clusters_meta.keys()):
            size = clusters_meta[cid]["size"]
            if size <= 5:
                target_cov = 0.90
            elif size <= 15:
                target_cov = 0.80
            else:
                target_cov = 0.70
            take = max(1, int(size * target_cov))
            print(f"       - Cluster {cid} ({size} items, cov={int(target_cov * 100)}%) -> take {take}")
            auto_n += take

        auto_n = max(10, auto_n)
        auto_n = min(auto_n, len(final_pool_indices))
        print(f"       -> Optimal: {auto_n} images")
        N = auto_n
    else:
        N = args.num_images
        if len(final_pool_indices) < N:
            raise ValueError(f"After filtering: {len(final_pool_indices)} images, need {N}.")

    C = len(clusters_meta)

    # ================================================================
    # GRID SEARCH: offsets -3..+3, 5 modes
    # ================================================================
    k_offsets = [-3, -2, -1, 0, 1, 2, 3]
    modes = ["proportional", "uniform", "sqrt", "log", "inverse"]

    print(f"[INFO] Grid search: {len(k_offsets) * len(modes)} strategies (K offsets: {k_offsets}, modes: {modes})")
    print(f"       Base clusters detected: {C}")

    candidates = []
    strategy_idx = 0

    valid_data_dedup = []
    for i, (gidx, path, size) in enumerate(valid_data_dedup_raw):
        valid_data_dedup.append({
            "global_idx": gidx,
            "path": path,
            "size": size,
            "emb": E_dedup[i]
        })

    # Кеш для KMeans и Silhouette Score
    kmeans_cache = {}

    for k_off in tqdm(k_offsets, desc="K-Offsets"):
        target_k = C + k_off
        target_k = max(2, min(target_k, args.max_clusters))
        target_k = min(target_k, N_valid // 2)
        if target_k < 2:
            target_k = 2

        if target_k not in kmeans_cache and N_valid >= 4:
            labels = KMeans(n_clusters=target_k, random_state=args.seed, n_init=10).fit_predict(E_dedup)
            ss = silhouette_score(E_dedup, labels)
            kmeans_cache[target_k] = {"labels": labels, "ss": ss}

        if target_k in kmeans_cache:
            current_labels = kmeans_cache[target_k]["labels"]
            current_ss = kmeans_cache[target_k]["ss"]

            new_clusters = {i: [] for i in range(target_k)}
            for idx, lbl in enumerate(current_labels):
                new_clusters[lbl].append(idx)

            new_valid_clusters = {cid: m for cid, m in new_clusters.items() if len(m) >= args.min_cluster_size}
            new_clusters_meta = {}
            new_final_pool = []
            for cid, members in new_valid_clusters.items():
                c_embs = E_dedup[members]
                center = np.mean(c_embs, axis=0)
                center /= np.linalg.norm(center)
                dists = 1.0 - (c_embs @ center)
                clean_members = [m for m, d in zip(members, dists) if
                                 d <= np.percentile(dists, args.outlier_percentile)]
                if len(clean_members) >= args.min_cluster_size:
                    new_clusters_meta[cid] = {
                        "size": len(clean_members),
                        "members": clean_members,
                        "center": center
                    }
                    new_final_pool.extend(clean_members)

            current_clusters_meta = new_clusters_meta
            current_pool_size = len(new_final_pool)
        else:
            current_clusters_meta = clusters_meta
            current_pool_size = len(final_pool_indices)
            current_ss = best_score

        current_C = len(current_clusters_meta)
        if current_C == 0:
            continue

        current_N = min(N, current_pool_size)
        if current_N < 2:
            continue

        for mode in modes:
            strategy_idx += 1
            result = build_selection_for_mode(
                current_clusters_meta, current_N, mode, valid_data_dedup,
                args.seed, strategy_idx, full_dist_matrix, current_ss
            )
            result["k_offset"] = k_off
            result["target_k"] = target_k
            result["actual_c"] = current_C
            result["strategy_idx"] = strategy_idx
            candidates.append(result)
            print(
                f"  #{strategy_idx:2d}  K={target_k:2d} (off={k_off:+d})  {mode:<13}  "
                f"N={current_N:3d}  C={current_C}  DS={result['ds']:.4f}  Gini={result['gini']:.4f}  "
                f"Cov={result['mean_coverage']:.1f}%  LRS={result['lrs']:.4f}  [{result['grade']}]  "
                f"dups={result['soft_dup_pairs']}"
            )

    if not candidates:
        raise ValueError("No valid strategies found after grid search.")

    candidates_sorted = sorted(
        candidates,
        key=lambda x: (x['lrs'], x['ds'], x['mean_coverage']),
        reverse=True
    )
    best = candidates_sorted[0]
    best['selected'] = True

    print(
        f"[INFO] BEST STRATEGY: #{best['strategy_idx']}  K={best['target_k']}  {best['mode'].upper()}  LRS={best['lrs']:.4f}")

    all_strategies = []
    for c in candidates:
        all_strategies.append({
            "strategy_idx": c["strategy_idx"],
            "k_offset": f"{c['k_offset']:+d} (K={c['target_k']})",
            "mode": c["mode"],
            "N": c["N"],
            "lrs": c["lrs"],
            "grade": c["grade"],
        })

    # Финальные метрики из лучшей стратегии
    ordered_selected_indices = best["ordered_indices"]
    sel_embs = best["sel_embs"]
    sel_global_indices = best["sel_global_indices"]
    path = best["path"]
    items_meta = best["items_meta"]
    dist_matrix = best["dist_matrix"]
    E_final = best["E_final"]
    best_clusters_meta = best["clusters_meta"]

    # Пересчитаем финальный QS с реальным SS для лучшей стратегии
    qs = calculate_quality_scores(best["N"], dist_matrix, best_clusters_meta, items_meta, best["ss_score"])

    # ИСПРАВЛЕНИЕ COVER IMAGE: используем отсортированный массив E_final
    print("[INFO] Selecting cover image...")
    ordered_global_indices = [sel_global_indices[i] for i in path]
    cover_idx, cover_dist = select_cover_image(E_final, ordered_global_indices)
    print(f"[INFO] Cover: NN-TSP index = {cover_idx}, distance to center = {cover_dist:.6f}")

    out_dir = os.path.join(args.input_dir, "selected")
    if os.path.exists(out_dir):
        if not args.force:
            raise FileExistsError(f"Folder {out_dir} exists. Use --force.")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    idx_to_cid = {m: cid for cid, meta in best_clusters_meta.items() for m in meta["members"]}

    cover_info = None
    final_items_meta = []

    # ИСПРАВЛЕНИЕ COVER IMAGE: i идет по ordered_selected_indices, что соответствует E_final
    for i, dedup_idx in enumerate(ordered_selected_indices):
        global_idx, orig_path, _ = valid_data_dedup_raw[dedup_idx]
        orig_name = Path(orig_path).name
        shutil.copy2(orig_path, os.path.join(out_dir, orig_name))
        cid = idx_to_cid.get(dedup_idx, -1)

        is_cover = (i == cover_idx)  # Теперь это сравнение корректно!
        item = {
            "index": i,
            "file": orig_name,
            "original_path": os.path.relpath(orig_path, args.input_dir),
            "cluster_id": int(cid),
            "cluster_size": int(best_clusters_meta[cid]["size"]) if cid != -1 else 0,
            "is_cover": is_cover
        }
        final_items_meta.append(item)

        if is_cover:
            ext = Path(orig_path).suffix
            cover_filename = f"cover{ext}"
            shutil.copy2(orig_path, os.path.join(out_dir, cover_filename))
            cover_info = {
                "file": cover_filename,
                "cluster_id": int(cid),
                "dist_to_center": cover_dist,
                "tsp_index": i,
                "original_path": os.path.relpath(orig_path, args.input_dir)
            }

    np.save(os.path.join(out_dir, "embeddings.npy"), E_final)
    np.save(os.path.join(out_dir, "distance_matrix.npy"), dist_matrix)

    stats = {
        "Total files found": total_found, "Broken files": broken_count,
        "Size filtered out": size_filtered_count, "Valid after filter": len(valid_data),
        "Duplicates removed": duplicates_removed, "Valid after dedup": N_valid,
        "Noise removed": noise_removed if N_valid >= 4 else 0,
        "Outliers removed": outliers_removed if N_valid >= 4 else 0,
        "Final pool size": len(final_pool_indices), "Target images (N)": best["N"],
        "Clusters found (C)": C, "N_selection_mode": "AUTO" if args.num_images is None else "MANUAL",
        "Selected_strategy": f"#{best['strategy_idx']} K={best['target_k']} {best['mode']}",
        "Cover image": cover_info["file"] if cover_info else "none"
    }

    with open(os.path.join(out_dir, "embeddings_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": {"version": "5.5-grid-optimized", "args": vars(args), "stats": stats, "cover": cover_info},
             "items": final_items_meta}, f, indent=2)

    benchmark_strategies = []
    for c in candidates:
        benchmark_strategies.append({
            "strategy_idx": c["strategy_idx"],
            "k_offset": f"{c['k_offset']:+d} (K={c['target_k']})",
            "mode": c["mode"],
            "N": c["N"],
            "ds": c.get("ds", 0),
            "gini": c.get("gini", 0),
            "mean_coverage": c.get("mean_coverage", 0),
            "lrs": c["lrs"],
            "grade": c["grade"],
            "selected": c.get("selected", False)
        })

    generate_benchmark(
        out_dir, args, stats, best_clusters_meta, dist_matrix,
        final_items_meta, qs, cover_info, benchmark_strategies, E_final.shape[1]
    )

    print_structured_summary(stats, best_clusters_meta, final_items_meta, qs, best, cover_info, all_strategies, args)

    print(f"[INFO] Result: {out_dir} ({best['N']} files + cover)")
    if qs['soft_dup_pairs'] > 0:
        print(
            f"[WARN] {qs['soft_dup_pairs']} soft duplicate pairs found (dist < {SOFT_DUP_THRESHOLD}). Consider tightening --dedup_threshold.")
    print(f"[INFO] LoRA Readiness Score (LRS): {qs['lrs']:.3f} ({qs['grade']})")
    if cover_info:
        print(f"[INFO] Dataset cover: {cover_info['file']}")


def main():
    parser = argparse.ArgumentParser(description="Dataset Benchmark Utility for LoRA (v5.5 Optimized)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=None,
                        help="If not set, auto-select by coverage heuristic")
    parser.add_argument("--min_size", type=int, default=512)
    parser.add_argument("--max_clusters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min_cluster_size", type=int, default=3)
    parser.add_argument("--dedup_threshold", type=float, default=0.99)
    parser.add_argument("--outlier_percentile", type=float, default=95.0)
    parser.add_argument("--device", type=str, default=None, help="Device for CLIP (cuda, cpu). Auto-detect if None.")
    parser.add_argument("--clip_model", type=str, default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
                        help="CLIP model name from HuggingFace.")
    parser.add_argument("--allow_synthetic_names", action="store_true",
                        help="Do not skip files with synthetic names like img_001.jpg")

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Directory {args.input_dir} not found.")

    run_pipeline(args)


if __name__ == "__main__":
    main()
