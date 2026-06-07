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

# Regex для запрета синтетических имен
SYNTHETIC_PATTERN = re.compile(r"^(item_|file_|img_|sample_)\d+", re.IGNORECASE)


def validate_filename(filename: str) -> bool:
    """Проверка имени файла на соответствие стандарту оригинальных basename."""
    if '/' in filename or '\\' in filename: return False
    if SYNTHETIC_PATTERN.match(filename): return False
    if not re.match(r"^.+\.\w+$", filename): return False
    # Проверка на дублирование расширений (image.jpeg.jpeg)
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
    """Расчет коэффициента Джини для оценки баланса кластеров."""
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
            # КРИТИЧЕСКАЯ ВАЛИДАЦИЯ ИМЕНИ
            if not validate_filename(f):
                raise ValueError(
                    f"КРИТИЧЕСКАЯ ОШИБКА: Файл '{f}' имеет синтетическое или невалидное имя. "
                    f"Требуется оригинальный basename. Остановка пайплайна."
                )
            all_paths.append(full_path)

    all_paths.sort()

    valid_data = []
    broken_count = 0
    size_filtered_count = 0

    print(f"[INFO] Сканирование (без рекурсии): найдено {len(all_paths)} файлов")
    for idx, path in enumerate(tqdm(all_paths, desc="Фильтрация")):
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
            print(f"[WARN] Битый файл: {path}")

    return valid_data, len(all_paths), broken_count, size_filtered_count


# ==============================================================================
# 3. ГЕНЕРАЦИЯ ОТЧЕТА (7 СЕКЦИЙ)
# ==============================================================================

def generate_benchmark(out_dir, args, stats, clusters_meta, dist_matrix, items_meta):
    report = []
    sep = "=" * 60

    report.append(sep)
    report.append("LORA DATASET BENCHMARK REPORT")
    report.append(sep)
    report.append(f"Timestamp: {datetime.now().isoformat()}")
    report.append(f"Utility Version: 4.0 (Strict Naming & Deep Benchmark)")
    report.append("")

    # [1. CONFIGURATION]
    report.append("[1. CONFIGURATION]")
    for k, v in vars(args).items():
        report.append(f"{k:<22}: {v}")
    report.append("")

    # [2. PIPELINE STATISTICS]
    report.append("[2. PIPELINE STATISTICS]")
    for k, v in stats.items():
        report.append(f"{k:<25}: {v}")
    report.append("")

    # [3. CLUSTER DISTRIBUTION]
    report.append("[3. CLUSTER DISTRIBUTION]")
    report.append(f"{'Cluster ID':<12} | {'Size':<8} | {'Selected':<10} | {'Coverage %':<10}")
    report.append("-" * 50)
    for cid, meta in sorted(clusters_meta.items()):
        sel_count = sum(1 for item in items_meta if item['cluster_id'] == cid)
        coverage = (sel_count / meta['size'] * 100) if meta['size'] > 0 else 0
        report.append(f"{cid:<12} | {meta['size']:<8} | {sel_count:<10} | {coverage:<10.2f}")
    report.append("")

    # [4. VECTOR SPACE METRICS]
    report.append("[4. VECTOR SPACE METRICS (Distance Matrix)]")
    N = dist_matrix.shape[0]
    if N > 1:
        i, j = np.triu_indices(N, k=1)
        upper_dists = dist_matrix[i, j]
        report.append(f"Mean pairwise distance : {np.mean(upper_dists):.6f}")
        report.append(f"Median pairwise dist.  : {np.median(upper_dists):.6f}")
        report.append(f"Min pairwise distance  : {np.min(upper_dists):.6f}")
        report.append(f"Max pairwise distance  : {np.max(upper_dists):.6f}")
        report.append(f"Std deviation          : {np.std(upper_dists):.6f}")
    report.append("")

    # [5. TSP PATH (FINAL ORDER)]
    report.append("[5. TSP PATH (FINAL ORDER)]")
    report.append(f"{'Index':<7} | {'File (basename)':<40} | {'Cluster':<8}")
    report.append("-" * 70)
    for item in items_meta:
        report.append(f"{item['index']:<7} | {item['file']:<40} | {item['cluster_id']:<8}")
    report.append("")

    # [6. FILE INTEGRITY]
    report.append("[6. FILE INTEGRITY (SHA-256)]")
    emb_path = os.path.join(out_dir, "embeddings.npy")
    dist_path = os.path.join(out_dir, "distance_matrix.npy")
    report.append(f"embeddings.npy      : {get_file_sha256(emb_path)}")
    report.append(f"distance_matrix.npy : {get_file_sha256(dist_path)}")
    report.append("")

    # [7. VISIONEMBED BENCHMARK]
    report.append(sep)
    report.append("VISIONEMBED BENCHMARK")
    report.append(sep)
    report.append(f"Dataset:   {args.input_dir}")
    report.append(f"Images:    {N}")
    report.append(f"Embedding: 768-d CLIP ViT-L/14")
    report.append(f"Seed:      {args.seed}")
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
        nn_dists = np.min(temp_matrix, axis=1)
        mean_nn_dist = np.mean(nn_dists)
        report.append(f"Mean nearest-neighbor dist: {mean_nn_dist:.4f}  (diversity)")

    selected_counts = {}
    for item in items_meta:
        cid = item['cluster_id']
        selected_counts[cid] = selected_counts.get(cid, 0) + 1

    gini = calculate_gini(list(selected_counts.values()))
    report.append(f"Cluster balance (Gini):     {gini:.4f}  (0=perfect, 1=worst)")
    report.append("")

    report.append("CLUSTER COMPOSITION")
    report.append("----------------------------------------")
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

    bench_path = os.path.join(out_dir, "benchmark.txt")
    with open(bench_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"[INFO] Эталонный отчет сохранен: {bench_path}")


# ==============================================================================
# 4. ОСНОВНОЙ ПАЙПЛАЙН
# ==============================================================================

def run_pipeline(args):
    set_global_seed(args.seed)

    valid_data, total_found, broken_count, size_filtered_count = load_and_filter_images(args.input_dir, args.min_size)
    if len(valid_data) < args.num_images:
        raise ValueError(f"Найдено {len(valid_data)} валидных изображений, но запрошено {args.num_images}.")

    print(f"[INFO] Загрузка модели CLIP...")
    device = "cpu"
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K", torch_dtype=torch.float32).to(device)
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
    model.eval()

    embeddings = []
    pbar = tqdm(total=len(valid_data), desc="Векторизация (файлы)")
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
                if hasattr(feats, 'pooler_output'):
                    feats = feats.pooler_output
                elif hasattr(feats, 'image_embeds'):
                    feats = feats.image_embeds
                else:
                    raise TypeError(f"Не удалось извлечь тензор из {type(feats)}")

            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())
            pbar.update(len(batch_data))
    pbar.close()

    E = np.vstack(embeddings).astype(np.float32)

    print(f"[INFO] Дедупликация (порог {args.dedup_threshold})...")
    N = E.shape[0]
    uf = UnionFind(N)
    chunk_size = 1024
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        for j in range(i, N, chunk_size):
            end_j = min(j + chunk_size, N)
            sim = E[i:end_i] @ E[j:end_j].T
            if i == j: np.fill_diagonal(sim, -1.0)
            rows, cols = np.where(sim >= args.dedup_threshold)
            for r, c in zip(rows, cols): uf.union(i + r, j + c)

    groups = {}
    for i in range(N):
        root = uf.find(i)
        if root not in groups: groups[root] = []
        groups[root].append(i)

    dedup_indices = []
    duplicates_removed = 0
    for members in groups.values():
        if len(members) > 1: duplicates_removed += len(members) - 1
        best_member = max(members, key=lambda m: (valid_data[m][2], -valid_data[m][0]))
        dedup_indices.append(best_member)

    dedup_indices.sort()
    E_dedup = E[dedup_indices]
    valid_data_dedup = [valid_data[i] for i in dedup_indices]
    print(f"[INFO] 🛡️ Дедупликация: удалено {duplicates_removed} копий. Осталось: {len(E_dedup)}")

    N_valid = len(E_dedup)
    final_pool_indices = []
    clusters_meta = {}
    noise_removed = 0
    outliers_removed = 0

    if N_valid < 4:
        print(f"[INFO] N_valid < 4. Пропуск кластеризации.")
        final_pool_indices = list(range(N_valid))
        clusters_meta[0] = {"size": N_valid, "members": final_pool_indices}
    else:
        max_k = min(args.max_clusters, N_valid // 2)
        if max_k < 2: max_k = 2
        best_k, best_score = 2, -1.0

        print(f"[INFO] Кластеризация (подбор k от 2 до {max_k})...")
        for k in tqdm(range(2, max_k + 1), desc="Подбор k (KMeans)"):
            kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
            labels = kmeans.fit_predict(E_dedup)
            score = silhouette_score(E_dedup, labels)
            if score > best_score: best_score, best_k = score, k

        print(f"[INFO] ✅ Выбрано k={best_k} (silhouette: {best_score:.3f})")
        kmeans = KMeans(n_clusters=best_k, random_state=args.seed, n_init=10)
        labels = kmeans.fit_predict(E_dedup)
        clusters = {i: [] for i in range(best_k)}
        for idx, lbl in enumerate(labels): clusters[lbl].append(idx)

        valid_clusters = {}
        for cid, members in clusters.items():
            if len(members) < args.min_cluster_size:
                noise_removed += len(members)
            else:
                valid_clusters[cid] = members

        print(f"[INFO] 🛡️ Шум: удалено {noise_removed} изображений из микро-кластеров.")

        for cid, members in tqdm(valid_clusters.items(), desc="Отсечение выбросов"):
            c_embs = E_dedup[members]
            center = np.mean(c_embs, axis=0)
            center /= np.linalg.norm(center)
            dists = 1.0 - (c_embs @ center)
            threshold = np.percentile(dists, args.outlier_percentile)
            clean_members = [m for m, d in zip(members, dists) if d <= threshold]
            outliers_removed += len(members) - len(clean_members)
            if len(clean_members) >= args.min_cluster_size:
                clusters_meta[cid] = {"size": len(clean_members), "members": clean_members}
                final_pool_indices.extend(clean_members)

        print(f"[INFO] 🛡️ Выбросы: удалено {outliers_removed} изображений (периферия).")

    if len(final_pool_indices) < args.num_images:
        raise ValueError(f"После фильтраций осталось {len(final_pool_indices)} изобр., нужно {args.num_images}.")

    N = args.num_images
    C = len(clusters_meta)
    selected_indices = []

    if N < C:
        sorted_cids = sorted(clusters_meta.keys(), key=lambda c: (-clusters_meta[c]["size"], c))
        for cid in sorted_cids[:N]:
            selected_indices.append(random.choice(clusters_meta[cid]["members"]))
    else:
        quotas = {cid: 1 for cid in clusters_meta}
        R = N - C
        total_valid = sum(meta["size"] for meta in clusters_meta.values())
        adds, rems = {}, {}
        for cid, meta in clusters_meta.items():
            share = R * (meta["size"] / total_valid)
            add = int(np.floor(share))
            adds[cid], rems[cid] = add, share - add
            quotas[cid] += add

        remaining_slots = R - sum(adds.values())
        if remaining_slots > 0:
            pool_embs = E_dedup[final_pool_indices]
            global_center = np.mean(pool_embs, axis=0)
            global_center /= np.linalg.norm(global_center)
            dists_to_global = {}
            for cid, meta in clusters_meta.items():
                c_center = np.mean(E_dedup[meta["members"]], axis=0)
                c_center /= np.linalg.norm(c_center)
                dists_to_global[cid] = 1.0 - np.dot(c_center, global_center)
            sorted_candidates = sorted(clusters_meta.keys(), key=lambda c: (quotas[c], -dists_to_global[c], c))
            for i in range(remaining_slots): quotas[sorted_candidates[i]] += 1

        for cid, meta in clusters_meta.items():
            if quotas[cid] > 0: selected_indices.extend(random.sample(meta["members"], quotas[cid]))

    sel_embs = E_dedup[selected_indices]
    sel_global_indices = [valid_data_dedup[i][0] for i in selected_indices]
    center = np.mean(sel_embs, axis=0)
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
        path.append(nxt)
        used.add(nxt)

    ordered_selected_indices = [selected_indices[i] for i in path]
    out_dir = os.path.join(args.input_dir, "selected")
    if os.path.exists(out_dir):
        if not args.force: raise FileExistsError(f"Папка {out_dir} уже существует. Используйте --force.")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    items_meta = []
    final_embs = []
    idx_to_cid = {m: cid for cid, meta in clusters_meta.items() for m in meta["members"]}

    for i, dedup_idx in enumerate(tqdm(ordered_selected_indices, desc="Сохранение (ориг. имена)")):
        global_idx, orig_path, _ = valid_data_dedup[dedup_idx]
        orig_name = Path(orig_path).name  # СТРОГО BASENAME

        shutil.copy2(orig_path, os.path.join(out_dir, orig_name))
        final_embs.append(sel_embs[path[i]])
        cid = idx_to_cid.get(dedup_idx, -1)
        items_meta.append({
            "index": i,
            "file": orig_name,
            "original_path": os.path.relpath(orig_path, args.input_dir),
            "cluster_id": int(cid),
            "cluster_size": int(clusters_meta[cid]["size"]) if cid != -1 else 0
        })

    E_final = np.vstack(final_embs).astype(np.float32)
    np.save(os.path.join(out_dir, "embeddings.npy"), E_final)

    dist_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(0, N, 512):
        chunk = E_final[i:min(i + 512, N)]
        dist_matrix[i:min(i + 512, N)] = np.clip(1.0 - (chunk @ E_final.T), 0.0, 2.0)
    np.fill_diagonal(dist_matrix, 0.0)
    np.save(os.path.join(out_dir, "distance_matrix.npy"), dist_matrix)

    stats = {
        "Total files found": total_found,
        "Broken files": broken_count,
        "Size filtered out": size_filtered_count,
        "Valid after filter": len(valid_data),
        "Duplicates removed": duplicates_removed,
        "Valid after dedup": N_valid,
        "Noise removed": noise_removed if N_valid >= 4 else 0,
        "Outliers removed": outliers_removed if N_valid >= 4 else 0,
        "Final pool size": len(final_pool_indices),
        "Target images (N)": args.num_images,
        "Clusters found (C)": C
    }

    meta_out = {
        "metadata": {"utility_version": "4.0", "timestamp": datetime.now().isoformat(), "args": vars(args),
                     "stats": stats},
        "items": items_meta
    }
    with open(os.path.join(out_dir, "embeddings_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    generate_benchmark(out_dir, args, stats, clusters_meta, dist_matrix, items_meta)
    print(f"[INFO] Результат: {out_dir} ({N} файлов с оригинальными именами)")


def main():
    parser = argparse.ArgumentParser(description="Dataset Benchmark Utility for LoRA")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--num_images", type=int, required=True)
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