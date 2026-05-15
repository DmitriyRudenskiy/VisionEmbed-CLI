#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# -----------------------------------------------------------------------------
# КОНСТАНТЫ И КОНФИГУРАЦИЯ ПО УМОЛЧАНИЮ
# -----------------------------------------------------------------------------
EPSILON = 1e-10
DEFAULT_RANDOM_SEED = 42
SUPPORTED_IMAGE_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'})
DEFAULT_CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_CLUSTERS = 20

# -----------------------------------------------------------------------------
# ЗАВИСИМОСТИ С ИНФОРМАТИВНЫМИ ОШИБКАМИ
# -----------------------------------------------------------------------------
try:
    import cv2
except ImportError as exc:
    raise ImportError("Требуется OpenCV. Установите: pip install opencv-python-headless") from exc

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
except ImportError as exc:
    raise ImportError("Требуется scikit-learn. Установите: pip install scikit-learn") from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment]

# Прогресс-бар (опционально)
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None  # type: ignore[assignment]

def _noop_progress_bar(iterable, **kwargs):
    return iterable

progress_bar = _tqdm if _tqdm is not None else _noop_progress_bar

logger = logging.getLogger(__name__)


# =============================================================================
# УТИЛИТЫ
# =============================================================================

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-нормализация для численной стабильности косинусных метрик."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + EPSILON)


def clip_similarity(values: np.ndarray) -> np.ndarray:
    """Ограничивает косинусное сходство диапазоном [0.0, 1.0]."""
    return np.clip(values, 0.0, 1.0)


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Преобразует изображение в RGB (HxWx3, uint8).
    Предполагается, что на вход подаётся BGR (стандарт OpenCV),
    либо серое/четырёхканальное изображение.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    # Трёхканальный случай: считаем, что это BGR
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_image_robust(path: Path) -> Optional[np.ndarray]:
    """Безопасная загрузка изображения (в т.ч. с кириллическими путями)."""
    try:
        raw_bytes = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        return img if img is not None and img.size > 0 else None
    except Exception as exc:
        logger.debug("Ошибка чтения %s: %s", path, exc)
        return None


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

@dataclass
class DiversitySelectionParams:
    """
    Параметры алгоритма Maximal Marginal Relevance (MMR).
    Формула: score = λ * relevance + (1 - λ) * (1 - max_similarity_to_selected)
    """
    relevance_diversity_balance: float = 0.5
    random_seed: int = DEFAULT_RANDOM_SEED

    def __post_init__(self):
        if not 0.0 <= self.relevance_diversity_balance <= 1.0:
            raise ValueError("relevance_diversity_balance должен быть в диапазоне [0.0, 1.0]")


@dataclass
class CurationPipelineConfig:
    """Полная конфигурация пайплайна кураторства датасета."""
    input_directory: str
    output_json_path: str
    reference_directory: Optional[str] = None

    target_subset_size: int = 100
    max_clusters: int = DEFAULT_MAX_CLUSTERS
    batch_size: int = DEFAULT_BATCH_SIZE

    mmr_config: DiversitySelectionParams = field(default_factory=DiversitySelectionParams)
    outlier_contamination: float = 0.1
    export_distance_matrix: bool = False

    clip_model_name: str = DEFAULT_CLIP_MODEL_NAME
    use_gpu: bool = True

    def __post_init__(self):
        if self.target_subset_size <= 0:
            raise ValueError("target_subset_size должен быть > 0")
        if not 0.0 < self.outlier_contamination <= 0.5:
            raise ValueError("outlier_contamination должен быть в диапазоне (0, 0.5]")


# =============================================================================
# ИНТЕРФЕЙСЫ (АБСТРАКЦИИ)
# =============================================================================

class EmbeddingExtractor(ABC):
    """Интерфейс извлечения нормализованных векторных представлений изображений."""
    @property
    @abstractmethod
    def embedding_dimension(self) -> int: ...

    @abstractmethod
    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        """Возвращает L2-нормализованные эмбеддинги (N, D)."""
        ...


class OutlierDetector(ABC):
    """Интерфейс детектора аномалий в пространстве эмбеддингов."""
    @abstractmethod
    def compute_inlier_mask(self, embeddings: np.ndarray) -> np.ndarray: ...


class ClusterAssigner(ABC):
    """Интерфейс назначения кластерных меток."""
    @abstractmethod
    def compute_cluster_labels(self, embeddings: np.ndarray, max_clusters: int) -> np.ndarray: ...


# =============================================================================
# РЕАЛИЗАЦИИ
# =============================================================================

class CLIPEmbeddingExtractor(EmbeddingExtractor):
    """Извлечение эмбеддингов через CLIP (sentence-transformers)."""
    _model_cache: dict[str, SentenceTransformer] = {}

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL_NAME,
        use_gpu: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        if SentenceTransformer is None:
            raise ImportError("Требуется sentence-transformers. Установите: pip install sentence-transformers")

        self.model_name = model_name
        self.batch_size = batch_size

        if model_name not in self._model_cache:
            logger.info("Загрузка CLIP-модели: %s", model_name)
            device = "cuda" if (TORCH_AVAILABLE and use_gpu and torch.cuda.is_available()) else "cpu"
            logger.info("Вычислительное устройство: %s", device)

            self._model_cache[model_name] = SentenceTransformer(
                model_name, device=device, trust_remote_code=True
            )

        self._model = self._model_cache[model_name]
        self._embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info("CLIP готов. Размерность эмбеддингов: %d", self._embedding_dim)

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dim

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        from PIL import Image
        pil_images = [
            Image.fromarray(convert_to_rgb(img)) if not isinstance(img, Image.Image) else img
            for img in images
        ]

        embeddings = self._model.encode(
            pil_images,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


class IsolationForestOutlierDetector(OutlierDetector):
    """Детектор выбросов на основе Isolation Forest."""
    def __init__(self, contamination: float = 0.1, random_seed: int = DEFAULT_RANDOM_SEED):
        if not 0.0 < contamination <= 0.5:
            raise ValueError("contamination должен быть в диапазоне (0, 0.5]")
        self.contamination = contamination
        self.random_seed = random_seed

    def compute_inlier_mask(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) < 10:
            logger.warning("Мало образцов для детекции выбросов (<10). Пропускаем фильтрацию.")
            return np.ones(len(embeddings), dtype=bool)

        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_seed,
            n_jobs=-1,
        )
        predictions = model.fit_predict(embeddings)
        return predictions == 1


class AdaptiveKMeansClusterer(ClusterAssigner):
    """K-Means с адаптивным выбором числа кластеров: max(2, min(max_k, sqrt(N/2)))."""
    def __init__(self, random_seed: int = DEFAULT_RANDOM_SEED):
        self.random_seed = random_seed

    def compute_cluster_labels(self, embeddings: np.ndarray, max_clusters: int) -> np.ndarray:
        n_samples = len(embeddings)
        if n_samples <= 2:
            return np.zeros(n_samples, dtype=int)

        estimated_k = int(np.sqrt(n_samples / 2))
        n_clusters = max(2, min(max_clusters, estimated_k))
        logger.debug("Кластеризация: %d образцов → %d кластеров", n_samples, n_clusters)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_seed,
            n_init=10,
            max_iter=300,
            n_jobs=-1,
        )
        return kmeans.fit_predict(embeddings)


# =============================================================================
# ЗАГРУЗКА И ЭКСПОРТ ДАННЫХ
# =============================================================================

def collect_image_paths(directory: str) -> list[Path]:
    """Рекурсивный поиск изображений с поддерживаемыми расширениями."""
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        raise ValueError(f"Директория не найдена: {dir_path}")

    paths = sorted(
        p for p in dir_path.rglob('*')
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and p.is_file()
    )
    logger.info("Найдено %d изображений в %s", len(paths), dir_path)
    return paths


def load_and_preprocess_images(paths: list[Path]) -> tuple[list[np.ndarray], list[Path], int]:
    """Пакетная загрузка и приведение к RGB."""
    images, valid_paths, failed_count = [], [], 0

    for p in paths:
        img = load_image_robust(p)
        if img is not None:
            images.append(convert_to_rgb(img))
            valid_paths.append(p)
        else:
            failed_count += 1

    if failed_count:
        logger.warning("Пропущено %d из %d изображений (битые/неподдерживаемые)", failed_count, len(paths))

    return images, valid_paths, failed_count


@dataclass
class CurationResult:
    """Контейнер результатов отбора с метаданными."""
    selected_paths: list[Path]
    relevance_scores: np.ndarray
    cluster_labels: np.ndarray
    original_embeddings: Optional[np.ndarray] = None

    @property
    def count(self) -> int:
        return len(self.selected_paths)

    def to_serializable_list(self) -> list[dict]:
        return [
            {
                "file_path": str(p),
                "relevance_score": float(score),
                "cluster_id": int(cluster),
            }
            for p, score, cluster in zip(self.selected_paths, self.relevance_scores, self.cluster_labels)
        ]


def save_selection_to_json(result: CurationResult, output_path: str) -> None:
    """Сохранение результатов в JSON со статистикой."""
    if not result.selected_paths:
        logger.warning("Пустой результат отбора. Экспорт пропущен.")
        return

    scores = result.relevance_scores
    relevance_stats = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
    }

    unique, counts = np.unique(result.cluster_labels, return_counts=True)
    cluster_distribution = {int(k): int(v) for k, v in zip(unique, counts)}

    payload = {
        "metadata": {
            "total_selected": result.count,
            "relevance_statistics": relevance_stats,
            "cluster_distribution": cluster_distribution,
            "embedding_dimension": result.original_embeddings.shape[1] if result.original_embeddings is not None else None,
        },
        "selection": result.to_serializable_list(),
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Результаты сохранены: %s (%d изображений)", out_file, result.count)


def save_distance_matrix(embeddings: np.ndarray, base_output_path: str) -> None:
    """Экспорт матрицы попарных косинусных расстояний (.npy)."""
    if len(embeddings) == 0:
        logger.warning("Пустые эмбеддинги. Матрица расстояний не создана.")
        return

    normalized = normalize_embeddings(embeddings)
    similarity_matrix = normalized @ normalized.T
    distance_matrix = 1.0 - clip_similarity(similarity_matrix)
    np.fill_diagonal(distance_matrix, 0.0)

    dist_path = Path(base_output_path).with_suffix('.npy')
    np.save(dist_path, distance_matrix)
    logger.info("Матрица расстояний %s сохранена: %s", distance_matrix.shape, dist_path)


# =============================================================================
# ЯДРО: MMR-ОТБОР
# =============================================================================

def mmr_select(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    relevance_scores: np.ndarray,
    config: DiversitySelectionParams,
    target_size: int,
) -> list[int]:
    """
    Жадный отбор по принципу Maximal Marginal Relevance.
    Гарантирует покрытие каждого кластера, затем добирает по маргинальной релевантности.
    Сложность: O(K·N·D), где K=target_size, N=кандидаты, D=размерность.
    """
    n_samples = len(embeddings)
    if n_samples == 0:
        return []
    if n_samples <= target_size:
        return list(range(n_samples))

    normalized = normalize_embeddings(embeddings)
    unique_clusters = np.unique(cluster_labels)

    # ШАГ 1: Гарантированное покрытие кластеров (лучший по релевантности в каждом)
    initial_selected: list[int] = []
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
            continue
        best_in_cluster = cluster_indices[np.argmax(relevance_scores[cluster_indices])]
        initial_selected.append(int(best_in_cluster))

    # Сортируем представителей по убыванию релевантности
    initial_selected.sort(key=lambda idx: relevance_scores[idx], reverse=True)

    # Булева маска выбранных элементов (быстрее set)
    is_selected = np.zeros(n_samples, dtype=bool)
    selected_count = 0
    for idx in initial_selected:
        if selected_count >= target_size:
            break
        if not is_selected[idx]:
            is_selected[idx] = True
            selected_count += 1

    # Инициализация максимального сходства с уже выбранными
    max_sim_to_selected = np.zeros(n_samples, dtype=np.float32)
    if selected_count > 0:
        selected_embeds = normalized[is_selected]
        sim_matrix = selected_embeds @ normalized.T  # (K, N)
        max_sim_to_selected = np.max(clip_similarity(sim_matrix), axis=0)

    # ШАГ 2: Жадный добор по маргинальной релевантности
    while selected_count < target_size:
        candidate_mask = ~is_selected
        candidate_indices = np.where(candidate_mask)[0]
        if len(candidate_indices) == 0:
            break

        rel_part = config.relevance_diversity_balance * relevance_scores[candidate_indices]
        div_part = (1.0 - config.relevance_diversity_balance) * (1.0 - max_sim_to_selected[candidate_indices])
        marginal_scores = rel_part + div_part

        best_local_idx = int(np.argmax(marginal_scores))
        best_global_idx = candidate_indices[best_local_idx]

        is_selected[best_global_idx] = True
        selected_count += 1

        # Обновляем максимальное сходство только для нового элемента
        new_sim = normalized @ normalized[best_global_idx]
        max_sim_to_selected = np.maximum(max_sim_to_selected, clip_similarity(new_sim))

    return np.where(is_selected)[0].tolist()


# =============================================================================
# ОРКЕСТРАТОР: ПАЙПЛАЙН КУРАТОРСТВА ДАТАСЕТА
# =============================================================================

class ImageCurationPipeline:
    """
    Полный пайплайн отбора изображений:
    загрузка → эмбеддинги → фильтрация выбросов → кластеризация → MMR → экспорт.
    """
    def __init__(
        self,
        extractor: EmbeddingExtractor,
        outlier_detector: OutlierDetector,
        cluster_assigner: ClusterAssigner,
        config: CurationPipelineConfig,
    ):
        self.extractor = extractor
        self.outlier_detector = outlier_detector
        self.cluster_assigner = cluster_assigner
        self.config = config

        logger.info(
            "Пайплайн инициализирован: target=%d, MMR.λ=%.2f, max_clusters=%d",
            config.target_subset_size,
            config.mmr_config.relevance_diversity_balance,
            config.max_clusters,
        )

    def _extract_embeddings_batched(self, paths: list[Path]) -> tuple[np.ndarray, list[Path]]:
        """Пакетное извлечение эмбеддингов с прогресс-индикацией."""
        if not paths:
            return np.empty((0, self.extractor.embedding_dimension)), []

        all_embeddings: list[np.ndarray] = []
        valid_paths: list[Path] = []

        batch_indices = range(0, len(paths), self.config.batch_size)
        batch_progress = progress_bar(batch_indices, desc="Извлечение эмбеддингов", unit="батч")

        for start_idx in batch_progress:
            batch_paths = paths[start_idx : start_idx + self.config.batch_size]
            images, batch_valid, _ = load_and_preprocess_images(batch_paths)

            if images:
                try:
                    batch_emb = self.extractor.encode_images(images)
                    all_embeddings.append(batch_emb)
                    valid_paths.extend(batch_valid)
                except Exception as exc:
                    logger.warning("Ошибка при обработке батча %d-%d: %s", start_idx, start_idx+len(batch_paths)-1, exc)

        if not all_embeddings:
            return np.empty((0, self.extractor.embedding_dimension)), []

        return np.vstack(all_embeddings), valid_paths

    def _compute_relevance_scores(
        self,
        embeddings: np.ndarray,
        reference_embeddings: Optional[np.ndarray],
    ) -> np.ndarray:
        """Релевантность: к референсам или к центроиду датасета."""
        if reference_embeddings is not None and len(reference_embeddings) > 0:
            raw_scores = np.max(
                cosine_similarity(embeddings, reference_embeddings),
                axis=1
            )
            logger.info("Релевантность вычислена относительно %d референсов", len(reference_embeddings))
        else:
            logger.info("Референсы не заданы. Используется центроид датасета.")
            centroid = embeddings.mean(axis=0, keepdims=True)
            raw_scores = cosine_similarity(embeddings, centroid).flatten()

        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s > min_s + EPSILON:
            return (raw_scores - min_s) / (max_s - min_s)
        return np.ones_like(raw_scores)

    def run(self) -> Optional[CurationResult]:
        """Запуск полного пайплайна с обработкой ошибок."""
        # 1. Поиск изображений
        try:
            image_paths = collect_image_paths(self.config.input_directory)
        except ValueError as exc:
            logger.error("Ошибка при поиске изображений: %s", exc)
            return None

        if not image_paths:
            logger.error("Нет изображений в %s", self.config.input_directory)
            return None

        # 2. Референсные эмбеддинги (если заданы)
        reference_embeddings: Optional[np.ndarray] = None
        if self.config.reference_directory:
            try:
                ref_paths = collect_image_paths(self.config.reference_directory)
                if ref_paths:
                    logger.info("Обработка %d референсных изображений...", len(ref_paths))
                    reference_embeddings, _ = self._extract_embeddings_batched(ref_paths)
            except Exception as exc:
                logger.error("Не удалось обработать референсные изображения: %s", exc)
                # Продолжаем без референсов
                reference_embeddings = None

        # 3. Эмбеддинги основного датасета
        logger.info("Обработка основного датасета: %d изображений", len(image_paths))
        try:
            dataset_embeddings, valid_paths = self._extract_embeddings_batched(image_paths)
        except Exception as exc:
            logger.error("Критическая ошибка при извлечении эмбеддингов: %s", exc)
            return None

        if len(valid_paths) == 0:
            logger.error("Не удалось загрузить ни одного валидного изображения.")
            return None

        # 4. Фильтрация выбросов
        if len(valid_paths) > self.config.target_subset_size:
            logger.info("Фильтрация выбросов (Isolation Forest)...")
            try:
                inlier_mask = self.outlier_detector.compute_inlier_mask(dataset_embeddings)
                n_removed = int(np.sum(~inlier_mask))
                if n_removed > 0:
                    logger.info("Удалено %d выбросов (%.1f%%)", n_removed, 100 * n_removed / len(inlier_mask))
                dataset_embeddings = dataset_embeddings[inlier_mask]
                valid_paths = [p for p, keep in zip(valid_paths, inlier_mask) if keep]
            except Exception as exc:
                logger.warning("Ошибка при фильтрации выбросов: %s. Пропускаем этап.", exc)

        # 5. Кластеризация
        cluster_labels = np.zeros(len(valid_paths), dtype=int)
        if len(valid_paths) > self.config.target_subset_size:
            logger.info("Кластеризация (Adaptive K-Means)...")
            try:
                cluster_labels = self.cluster_assigner.compute_cluster_labels(
                    dataset_embeddings, max_clusters=self.config.max_clusters
                )
            except Exception as exc:
                logger.warning("Ошибка при кластеризации: %s. Кластерные метки будут нулевыми.", exc)

        # 6. Расчёт релевантности
        logger.info("Расчёт метрик релевантности...")
        try:
            relevance_scores = self._compute_relevance_scores(dataset_embeddings, reference_embeddings)
        except Exception as exc:
            logger.error("Ошибка при вычислении релевантности: %s", exc)
            return None

        # 7. MMR-отбор
        if len(valid_paths) > self.config.target_subset_size:
            logger.info("MMR-отбор: %d → %d", len(valid_paths), self.config.target_subset_size)
            try:
                selected_indices = mmr_select(
                    embeddings=dataset_embeddings,
                    cluster_labels=cluster_labels,
                    relevance_scores=relevance_scores,
                    config=self.config.mmr_config,
                    target_size=self.config.target_subset_size,
                )
                dataset_embeddings = dataset_embeddings[selected_indices]
                valid_paths = [valid_paths[i] for i in selected_indices]
                relevance_scores = relevance_scores[selected_indices]
                cluster_labels = cluster_labels[selected_indices]
            except Exception as exc:
                logger.error("Ошибка в процессе MMR-отбора: %s", exc)
                return None

        # 8. Экспорт
        try:
            if self.config.export_distance_matrix:
                save_distance_matrix(dataset_embeddings, self.config.output_json_path)

            result = CurationResult(
                selected_paths=valid_paths,
                relevance_scores=relevance_scores,
                cluster_labels=cluster_labels,
                original_embeddings=dataset_embeddings,
            )
            save_selection_to_json(result, self.config.output_json_path)
            logger.info("✅ Пайплайн завершён. Отобрано %d изображений.", result.count)
            return result
        except Exception as exc:
            logger.error("Ошибка при экспорте результатов: %s", exc)
            return None


# =============================================================================
# CLI: КОНСОЛЬНЫЙ ИНТЕРФЕЙС
# =============================================================================

def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🎨 Dataset Curation Pipeline — интеллектуальный отбор изображений для LoRA/fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Директория с исходными изображениями")
    parser.add_argument("--target-size", type=int, required=True, help="Желаемое количество изображений в выборке")
    parser.add_argument("--output-json", type=str, default="selection_result.json", help="Путь к выходному JSON")
    parser.add_argument("--reference-dir", type=str, default=None, help="Директория с референсными изображениями")
    parser.add_argument("--relevance-weight", type=float, default=0.5, help="Вес релевантности в MMR (0.0–1.0)")
    parser.add_argument("--max-clusters", type=int, default=DEFAULT_MAX_CLUSTERS, help="Максимальное число кластеров")
    parser.add_argument("--contamination", type=float, default=0.1, help="Ожидаемая доля выбросов")
    parser.add_argument("--clip-model", type=str, default=DEFAULT_CLIP_MODEL_NAME, help="Имя CLIP-модели")
    parser.add_argument("--cpu-only", action="store_true", help="Принудительно использовать CPU")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Размер батча для энкодера")
    parser.add_argument("--save-dist-matrix", action="store_true", help="Сохранить матрицу расстояний (.npy)")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed для воспроизводимости")
    parser.add_argument("--verbose", "-v", action="store_true", help="Включить отладочное логирование")
    return parser.parse_args()


def main() -> int:
    args = parse_cli_arguments()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        logger.error("❌ Директория не найдена: %s", input_path)
        return 1

    config = CurationPipelineConfig(
        input_directory=args.input_dir,
        output_json_path=args.output_json,
        reference_directory=args.reference_dir,
        target_subset_size=args.target_size,
        max_clusters=args.max_clusters,
        batch_size=args.batch_size,
        mmr_config=DiversitySelectionParams(
            relevance_diversity_balance=args.relevance_weight,
            random_seed=args.seed
        ),
        outlier_contamination=args.contamination,
        export_distance_matrix=args.save_dist_matrix,
        clip_model_name=args.clip_model,
        use_gpu=not args.cpu_only,
    )

    try:
        extractor = CLIPEmbeddingExtractor(
            model_name=config.clip_model_name,
            use_gpu=config.use_gpu,
            batch_size=config.batch_size,
        )
    except ImportError as exc:
        logger.error("❌ Ошибка инициализации CLIP: %s", exc)
        logger.info("💡 Установите: pip install sentence-transformers torch torchvision")
        return 1

    outlier_detector = IsolationForestOutlierDetector(
        contamination=config.outlier_contamination,
        random_seed=config.mmr_config.random_seed,
    )
    cluster_assigner = AdaptiveKMeansClusterer(random_seed=config.mmr_config.random_seed)

    pipeline = ImageCurationPipeline(
        extractor=extractor,
        outlier_detector=outlier_detector,
        cluster_assigner=cluster_assigner,
        config=config,
    )

    result = pipeline.run()
    return 0 if result is not None else 1


if __name__ == "__main__":
    sys.exit(main())