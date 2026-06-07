import argparse
import os
import sys
import json
import random
import logging
import shutil
import hashlib
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==============================================================================
# 1. НАСТРОЙКА ЛОГИРОВАНИЯ И ВАЛИДАЦИИ
# ==============================================================================

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

SYNTHETIC_PATTERN = re.compile(r"^(item_|file_|img_|sample_)\d+", re.IGNORECASE)


def validate_filename(filename: str) -> bool:
    if '/' in filename or '\\' in filename: return False
    if SYNTHETIC_PATTERN.match(filename): return False
    if not re.match(r"^.+\.\w+$", filename): return False
    parts = filename.split('.')
    if len(parts) > 2 and parts[-1].lower() == parts[-2].lower(): return False
    return True


def set_global_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()


def calculate_gini(sizes):
    if not sizes or sum(sizes) == 0: return 0.0
    n = len(sizes)
    sizes = np.array(sorted(sizes), dtype=float)
    index = np.arange(1, n + 1)
    return np.sum((2 * index - n - 1) * sizes) / (n * np.sum(sizes))


# ==============================================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ
# ==============================================================================

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j: self.parent[root_i] = root_j


def load_and_filter_images(input_dir: str, min_size: int):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
    all_paths = []

    entries = os.listdir(input_dir)
    for f in entries:
        full_path = os.path.abspath(os.path.join(input_dir, f))
        if os.path.isfile(full_path) and Path(f).suffix.lower() in valid_extensions:
            if not validate_filename(f):
                raise ValueError(
                    f"КРИТИЧЕСКАЯ ОШИБКА: Файл '{f}' имеет синтетическое имя. Требуется оригинальный basename.")
            all_paths.append(full_path)

    all_paths.sort()
    valid_data, broken_count, size_filtered_count = [], 0, 0

    print(f"[INFO] Сканирование (без рекурсии): найдено {len(all_paths)} файлов")
    for idx, path in enumerate(tqdm(all_paths, desc="Фильтрация")):
        try:
            with Image.open(path) as img:
                w, h = img.size
                if min(w, h) < min_size:
                    size_filtered_count += 1;
                    continue
                img.convert("RGB").load()
            valid_data.append((idx, path, os.path.getsize(path)))
        except Exception:
            broken_count += 1

    return valid_data, len(all_paths), broken_count, size_filtered_count


# ==============================================================================
# 3. РАСЧЁТ МЕТРИК КАЧЕСТВА (v5.0)
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
        soft_dup_pairs = int(np.sum(upper_dists < 0.025))

        ds = (mean_pairwise * mean_nn) / std_pairwise if std_pairwise > 0 else 0.0

    gini = calculate_gini(list(selected_counts.values()))

    norm_ds = min(ds / 0.35, 1.0)
    norm_gini = 1.0 - gini
    norm_cov = mean_coverage / 100.0
    norm_ss = max(0.0, min(ss_score, 1.0))

    lrs = 0.3 * norm_ds + 0.25 * norm_gini + 0.25 * norm_cov + 0.2 * norm_ss

    if mean_coverage < 60.0:
        lrs *= 0.7
    elif mean_coverage < 70.0:
        lrs *= 0.85

    if intra_min_dist < 0.025 and N > 1: lrs *= 0.9

    if lrs >= 0.80:
        grade = "🏆 GOOD"
    elif lrs >= 0.65:
        grade = "🟡 ACCEPTABLE"
    else:
        grade = "🔴 POOR"

    return {
        "ds": ds, "ce": ce, "ss": ss_score, "gini": gini,
        "mean_coverage": mean_coverage, "intra_min_dist": intra_min_dist,
        "soft_dup_pairs": soft_dup_pairs, "lrs": lrs, "grade": grade
    }


# ==============================================================================
# 4. ВЫБОР ОБЛОЖКИ (COVER IMAGE) — v5.2
# ==============================================================================

def select_cover_image(sel_embs, sel_global_indices):
    """
    Выбирает изображение, максимально близкое к центру масс всех отобранных.
    Возвращает индекс в sel_embs и расстояние до центра.
    """
    N = sel_embs.shape[0]
    if N == 0:
        return None, None

    # Центр масс (сферический центроид)
    centroid = np.mean(sel_embs, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    else:
        centroid = np.zeros_like(centroid)

    # Расстояния до центра
    dists_to_center = 1.0 - (sel_embs @ centroid)
    min_dist = np.min(dists_to_center)

    # Tie-breaker: при равенстве расстояний — меньший глобальный индекс
    candidates = np.where(np.abs(dists_to_center - min_dist) < 1e-6)[0]
    global_indices_array = np.array(sel_global_indices)
    cover_idx = candidates[np.argmin(global_indices_array[candidates])]

    return int(cover_idx), float(min_dist)


# ==============================================================================
# 5. ГЕНЕРАЦИЯ ОТЧЕТА
# ==============================================================================

def generate_benchmark(out_dir, args, stats, clusters_meta, dist_matrix, items_meta, qs, cover_info):
    report = []
    sep = "=" * 60
    N = dist_matrix.shape[0]

    report.append(sep)
    report.append("LORA DATASET BENCHMARK REPORT")
    report.append(sep)
    report.append(f"Timestamp: {datetime.now().isoformat()}")
    report.append(f"Utility Version: 5.2 (Cover Image Selection)")
    report.append("")

    report.append("[1. CONFIGURATION]")
    for k, v in vars(args).items(): report.append(f"{k:<22}: {v}")
    report.append("")

    report.append("[2. PIPELINE STATISTICS]")
    for k, v in stats.items(): report.append(f"{k:<25}: {v}")
    report.append("")

    report.append("[3. CLUSTER DISTRIBUTION]")
    report.append(f"{'Cluster ID':<12} | {'Size':<8} | {'Selected':<10} | {'Coverage %':<10}")
    report.append("-" * 50)
    for cid, meta in sorted(clusters_meta.items()):
        sel_count = sum(1 for item in items_meta if item['cluster_id'] == cid)
        coverage = (sel_count / meta['size'] * 100) if meta['size'] > 0 else 0
        report.append(f"{cid:<12} | {meta['size']:<8} | {sel_count:<10} | {coverage:<10.2f}")
    report.append("")

    report.append("[4. VECTOR SPACE METRICS (Distance Matrix)]")
    if N > 1:
        i, j = np.triu_indices(N, k=1)
        upper_dists = dist_matrix[i, j]
        report.append(f"Mean pairwise distance : {np.mean(upper_dists):.6f}")
        report.append(f"Median pairwise dist.  : {np.median(upper_dists):.6f}")
        report.append(f"Min pairwise distance  : {np.min(upper_dists):.6f}")
        report.append(f"Max pairwise distance  : {np.max(upper_dists):.6f}")
        report.append(f"Std deviation          : {np.std(upper_dists):.6f}")
    report.append("")

    report.append("[5. QUALITY SCORES v5.0]")
    report.append(f"Diversity Score (DS)       : {qs['ds']:.4f}")
    report.append(f"Coverage Entropy (CE)      : {qs['ce']:.4f}")
    report.append(f"Silhouette Score (SS)      : {qs['ss']:.4f}")
    report.append(f"Cluster Balance (Gini)     : {qs['gini']:.4f}")
    report.append(f"Mean Coverage              : {qs['mean_coverage']:.2f}%")

    min_dist_str = f"{qs['intra_min_dist']:.4f}"
    if qs['intra_min_dist'] < 0.025 and N > 1: min_dist_str += " ⚠️"
    report.append(f"Intra-cluster Min Distance : {min_dist_str}")

    dup_str = f"{qs['soft_dup_pairs']} pairs"
    if qs['soft_dup_pairs'] > 0: dup_str += " ⚠️"
    report.append(f"Soft Duplicates Found      : {dup_str}")

    report.append(f"LoRA Readiness Score (LRS) : {qs['lrs']:.4f} [{qs['grade']}]")
    report.append("")

    # [6. COVER IMAGE] — новая секция
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

    # [7. TSP PATH] — с маркером (COVER)
    report.append("[7. TSP PATH (FINAL ORDER)]")
    report.append(f"{'Index':<7} | {'File (basename)':<40} | {'Cluster':<8} | Note")
    report.append("-" * 80)
    for item in items_meta:
        note = "(COVER)" if item.get('is_cover') else ""
        report.append(f"{item['index']:<7} | {item['file']:<40} | {item['cluster_id']:<8} | {note}")
    report.append("")

    report.append("[8. FILE INTEGRITY (SHA-256)]")
    emb_path = os.path.join(out_dir, "embeddings.npy")
    dist_path = os.path.join(out_dir, "distance_matrix.npy")
    report.append(f"embeddings.npy      : {get_file_sha256(emb_path)}")
    report.append(f"distance_matrix.npy : {get_file_sha256(dist_path)}")
    if cover_info:
        cover_path = os.path.join(out_dir, cover_info['file'])
        if os.path.exists(cover_path):
            report.append(f"{cover_info['file']:<20}: {get_file_sha256(cover_path)}")
    report.append("")

    report.append(sep)
    report.append("VISIONEMBED BENCHMARK")
    report.append(sep)
    report.append(f"Dataset:   {args.input_dir}")
    report.append(f"Images:    {N}")
    report.append(f"Embedding: 768-d CLIP ViT-L/14")
    report.append(f"Seed:      {args.seed}")
    if cover_info:
        report.append(f"Cover:     {cover_info['file']}")
    report.append("")

    report.append("METRICS (cosine distance, 1 - similarity)")
    report.append("----------------------------------------")
    if N > 1:
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

    report.append("DISTANCE MATRIX")
    report.append("----------------------------------------")
    names = [item['file'] for item in items_meta]
    if names:
        max_len = max(len(n) for n in names)
        col_width = max(max_len, 10) + 2
        header = " " * col_width
        for n in names: header += f"{n:<{col_width}}"
        report.append(header)
        for i, n_row in enumerate(names):
            row_str = f"{n_row:<{col_width}}"
            for j in range(N): row_str += f"{dist_matrix[i, j]:<{col_width}.4f}"
            report.append(row_str)

    with open(os.path.join(out_dir, "benchmark.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))


# ==============================================================================
# 6. ОСНОВНОЙ ПАЙПЛАЙН
# ==============================================================================

def run_pipeline(args):
    set_global_seed(args.seed)
    valid_data, total_found, broken_count, size_filtered_count = load_and_filter_images(args.input_dir, args.min_size)
    if args.num_images is not None and len(valid_data) < args.num_images:
        raise ValueError(f"Найдено {len(valid_data)} валидных изображений, но запрошено {args.num_images}.")

    device = "cpu"
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K", torch_dtype=torch.float32).to(device)
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
    model.eval()

    embeddings = []
    pbar = tqdm(total=len(valid_data), desc="Векторизация")
    with torch.no_grad():
        for i in range(0, len(valid_data), args.batch_size):
            batch_data = valid_data[i:i + args.batch_size]
            images = [Image.open(d[1]).convert("RGB") for d in batch_data]
            inputs = processor(images=images, return_tensors="pt").to(device)
            try:
                feats = model.get_image_features(**inputs)
            except Exception:
                vision_outputs = model.vision_model(**inputs)
                feats = model.visual_projection(vision_outputs.pooler_output)
            if not isinstance(feats, torch.Tensor):
                feats = getattr(feats, 'pooler_output', getattr(feats, 'image_embeds', None))
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())
            pbar.update(len(batch_data))
    pbar.close()
    E = np.vstack(embeddings).astype(np.float32)

    # Дедупликация
    N = E.shape[0]
    uf = UnionFind(N)
    for i in range(0, N, 1024):
        for j in range(i, N, 1024):
            sim = E[i:min(i + 1024, N)] @ E[j:min(j + 1024, N)].T
            if i == j: np.fill_diagonal(sim, -1.0)
            rows, cols = np.where(sim >= args.dedup_threshold)
            for r, c in zip(rows, cols): uf.union(i + r, j + c)

    groups = {}
    for i in range(N):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)

    dedup_indices = [max(m, key=lambda x: (valid_data[x][2], -valid_data[x][0])) for m in groups.values()]
    dedup_indices.sort()
    E_dedup, valid_data_dedup = E[dedup_indices], [valid_data[i] for i in dedup_indices]
    duplicates_removed = N - len(E_dedup)

    N_valid = len(E_dedup)
    final_pool_indices, clusters_meta, noise_removed, outliers_removed, best_score = [], {}, 0, 0, 0.0

    if N_valid < 4:
        final_pool_indices = list(range(N_valid))
        clusters_meta[0] = {"size": N_valid, "members": final_pool_indices}
    else:
        max_k = min(args.max_clusters, N_valid // 2)
        if max_k < 2: max_k = 2
        best_k = 2
        for k in range(2, max_k + 1):
            labels = KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit_predict(E_dedup)
            score = silhouette_score(E_dedup, labels)
            if score > best_score: best_score, best_k = score, k

        labels = KMeans(n_clusters=best_k, random_state=args.seed, n_init=10).fit_predict(E_dedup)
        clusters = {i: [] for i in range(best_k)}
        for idx, lbl in enumerate(labels): clusters[lbl].append(idx)

        valid_clusters = {cid: m for cid, m in clusters.items() if len(m) >= args.min_cluster_size}
        noise_removed = sum(len(m) for cid, m in clusters.items() if len(m) < args.min_cluster_size)

        for cid, members in valid_clusters.items():
            c_embs = E_dedup[members]
            center = np.mean(c_embs, axis=0);
            center /= np.linalg.norm(center)
            dists = 1.0 - (c_embs @ center)
            clean_members = [m for m, d in zip(members, dists) if d <= np.percentile(dists, args.outlier_percentile)]
            outliers_removed += len(members) - len(clean_members)
            if len(clean_members) >= args.min_cluster_size:
                clusters_meta[cid] = {"size": len(clean_members), "members": clean_members}
                final_pool_indices.extend(clean_members)

    if len(final_pool_indices) == 0:
        raise ValueError("После всех фильтраций пул пуст. Невозможно сформировать датасет.")

    # 🧮 АВТОПОДБОР N (Target Coverage Method)
    if args.num_images is None:
        target_coverage = 0.75
        print(f"[INFO] 🧮 Автоподбор N (target coverage {int(target_coverage * 100)}%):")
        auto_n = 0
        for cid in sorted(clusters_meta.keys()):
            meta = clusters_meta[cid]
            take = max(1, int(meta["size"] * target_coverage))
            print(f"       - Кластер {cid} ({meta['size']} шт.) → взять {take}")
            auto_n += take

        auto_n = max(10, auto_n)
        auto_n = min(auto_n, len(final_pool_indices))

        print(f"       → Оптимально: {auto_n} изображений")
        N = auto_n
    else:
        N = args.num_images
        if len(final_pool_indices) < N:
            raise ValueError(f"После фильтраций осталось {len(final_pool_indices)} изобр., нужно {N}.")

    C = len(clusters_meta)
    selected_indices = []

    if N < C:
        for cid in sorted(clusters_meta.keys(), key=lambda c: (-clusters_meta[c]["size"], c))[:N]:
            selected_indices.append(random.choice(clusters_meta[cid]["members"]))
    else:
        quotas = {cid: 1 for cid in clusters_meta}
        R = N - C
        total_valid = sum(meta["size"] for meta in clusters_meta.values())
        adds = {cid: int(np.floor(R * (meta["size"] / total_valid))) for cid, meta in clusters_meta.items()}
        for cid in clusters_meta: quotas[cid] += adds[cid]

        remaining_slots = R - sum(adds.values())
        if remaining_slots > 0:
            pool_embs = E_dedup[final_pool_indices]
            global_center = np.mean(pool_embs, axis=0);
            global_center /= np.linalg.norm(global_center)
            dists_to_global = {cid: 1.0 - np.dot(
                np.mean(E_dedup[meta["members"]], axis=0) / np.linalg.norm(np.mean(E_dedup[meta["members"]], axis=0)),
                global_center) for cid, meta in clusters_meta.items()}
            for i, cid in enumerate(sorted(clusters_meta.keys(), key=lambda c: (quotas[c], -dists_to_global[c], c))):
                if i < remaining_slots: quotas[cid] += 1

        for cid, meta in clusters_meta.items():
            if quotas[cid] > 0: selected_indices.extend(random.sample(meta["members"], quotas[cid]))

    sel_embs = E_dedup[selected_indices]
    sel_global_indices = [valid_data_dedup[i][0] for i in selected_indices]
    center = np.mean(sel_embs, axis=0);
    center /= np.linalg.norm(center)
    dists_to_center = 1.0 - (sel_embs @ center)
    candidates = np.where(dists_to_center == np.max(dists_to_center))[0]
    path = [candidates[np.argmin(np.array(sel_global_indices)[candidates])]]
    used = {path[0]}

    for _ in range(N - 1):
        dists = 1.0 - (sel_embs @ sel_embs[path[-1]])
        dists[list(used)] = np.inf
        candidates = np.where(dists == np.min(dists))[0]
        nxt = candidates[np.argmin(np.array(sel_global_indices)[candidates])]
        path.append(nxt);
        used.add(nxt)

    ordered_selected_indices = [selected_indices[i] for i in path]

    # 🎨 ВЫБОР ОБЛОЖКИ (Cover Image)
    print(f"[INFO] 🎨 Выбор обложки (cover image)...")
    cover_idx, cover_dist = select_cover_image(sel_embs, sel_global_indices)
    print(f"[INFO] 🎨 Обложка: индекс в TSP = {cover_idx}, расстояние до центра = {cover_dist:.6f}")

    out_dir = os.path.join(args.input_dir, "selected")
    if os.path.exists(out_dir):
        if not args.force: raise FileExistsError(f"Папка {out_dir} существует. Используйте --force.")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    items_meta, final_embs = [], []
    idx_to_cid = {m: cid for cid, meta in clusters_meta.items() for m in meta["members"]}

    cover_info = None

    for i, dedup_idx in enumerate(ordered_selected_indices):
        global_idx, orig_path, _ = valid_data_dedup[dedup_idx]
        orig_name = Path(orig_path).name
        shutil.copy2(orig_path, os.path.join(out_dir, orig_name))
        final_embs.append(sel_embs[path[i]])
        cid = idx_to_cid.get(dedup_idx, -1)

        is_cover = (i == cover_idx)
        item = {
            "index": i,
            "file": orig_name,
            "original_path": os.path.relpath(orig_path, args.input_dir),
            "cluster_id": int(cid),
            "cluster_size": int(clusters_meta[cid]["size"]) if cid != -1 else 0,
            "is_cover": is_cover
        }
        items_meta.append(item)

        if is_cover:
            # Создаём копию с именем cover.ext
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

    E_final = np.vstack(final_embs).astype(np.float32)
    np.save(os.path.join(out_dir, "embeddings.npy"), E_final)

    dist_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(0, N, 512):
        chunk = E_final[i:min(i + 512, N)]
        dist_matrix[i:min(i + 512, N)] = np.clip(1.0 - (chunk @ E_final.T), 0.0, 2.0)
    np.fill_diagonal(dist_matrix, 0.0)
    np.save(os.path.join(out_dir, "distance_matrix.npy"), dist_matrix)

    qs = calculate_quality_scores(N, dist_matrix, clusters_meta, items_meta, best_score)

    stats = {
        "Total files found": total_found, "Broken files": broken_count,
        "Size filtered out": size_filtered_count, "Valid after filter": len(valid_data),
        "Duplicates removed": duplicates_removed, "Valid after dedup": N_valid,
        "Noise removed": noise_removed if N_valid >= 4 else 0,
        "Outliers removed": outliers_removed if N_valid >= 4 else 0,
        "Final pool size": len(final_pool_indices), "Target images (N)": N,
        "Clusters found (C)": C, "N_selection_mode": "AUTO" if args.num_images is None else "MANUAL",
        "Cover image": cover_info["file"] if cover_info else "none"
    }

    with open(os.path.join(out_dir, "embeddings_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"version": "5.2", "args": vars(args), "stats": stats, "cover": cover_info},
                   "items": items_meta}, f, indent=2)

    generate_benchmark(out_dir, args, stats, clusters_meta, dist_matrix, items_meta, qs, cover_info)

    print(f"[INFO] Результат: {out_dir} ({N} файлов + обложка)")
    if qs['soft_dup_pairs'] > 0:
        print(
            f"[WARN] ⚠️ Обнаружено {qs['soft_dup_pairs']} пар soft-дубликатов (dist < 0.025). Рекомендуется ужесточить --dedup_threshold.")
    print(f"[INFO] 📊 LoRA Readiness Score (LRS): {qs['lrs']:.3f} ({qs['grade']})")
    if cover_info:
        print(f"[INFO] 🎨 Обложка датасета: {cover_info['file']}")


def main():
    parser = argparse.ArgumentParser(description="Dataset Benchmark Utility for LoRA (v5.2)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=None,
                        help="Если не указан, автоподбор по эвристике 75%% coverage")
    parser.add_argument("--min_size", type=int, default=512)
    parser.add_argument("--max_clusters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min_cluster_size", type=int, default=3)
    parser.add_argument("--dedup_threshold", type=float, default=0.99)
    parser.add_argument("--outlier_percentile", type=float, default=95.0)
    args = parser.parse_args()
    if not os.path.isdir(args.input_dir): raise FileNotFoundError(f"Директория {args.input_dir} не найдена.")
    run_pipeline(args)


if __name__ == "__main__":
    main()