#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Selection Pipeline — система интеллектуального отбора изображений
для обучения LoRA/текстовых моделей.

Использует CLIP-эмбеддинги, MMR-отбор и кластеризацию для формирования
сбалансированного и релевантного датасета.

Author: Your Name
Date: 2026
"""

from __future__ import annotations

import json
import argparse
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union
from contextlib import contextmanager

import numpy as np

# -----------------------------------------------------------------------------
# Константы и конфигурация
# -----------------------------------------------------------------------------
EPSILON = 1e-10
DEFAULT_RANDOM_STATE = 42
SUPPORTED_IMAGE_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'})
DEFAULT_CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
BATCH_SIZE_DEFAULT = 32
MAX_CLUSTERS_DEFAULT = 20

# -----------------------------------------------------------------------------
# Зависимости с информативными ошибками
# -----------------------------------------------------------------------------
try:
    import cv2
except ImportError as e:
    raise ImportError(
        "Требуется OpenCV. Установите: pip install opencv-python-headless"
    ) from e

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
except ImportError as e:
    raise ImportError(
        "Требуется scikit-learn. Установите: pip install scikit-learn"
    ) from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

logger = logging.getLogger(__name__)


# =============================================================================
# УТИЛИТЫ
# =============================================================================

@contextmanager
def suppress_tqdm_if_needed(use_progress: bool = True):
    """Контекстный менеджер для условного отключения tqdm."""
    if use_progress and tqdm is not None:
        yield tqdm
    else:
        # Заглушка-итератор
        def dummy_tqdm(iterable=None, *args, **kwargs):
            return iterable if iterable is not None else []
        yield dummy_tqdm


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-нормализация эмбеддингов с численной стабильностью."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + EPSILON)


def _clip_similarity(values: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """Безопасное ограничение значений косинусного сходства."""
    return np.clip(values, min_val, max_val)


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Гарантирует, что изображение в формате RGB (HxWx3, uint8)."""
    if len(image.shape) == 2:  # Grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:  # RGBA
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.flags.writeable else image.copy()
    raise ValueError(f"Неподдерживаемый формат изображения: {image.shape}")


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

@dataclass
class DiversityRelevanceConfig:
    """
    Баланс между релевантностью и разнообразием в алгоритме MMR.

    Формула маргинальной релевантности:
        score = λ * relevance + (1 - λ) * (1 - max_similarity_to_selected)

    Attributes:
        relevance_weight: λ ∈ [0, 1]. 1.0 = только релевантность, 0.0 = только разнообразие.
    """
    relevance_weight: float = 0.5

    def __post_init__(self):
        if not 0.0 <= self.relevance_weight <= 1.0:
            raise ValueError("relevance_weight должен быть в диапазоне [0.0, 1.0]")


@dataclass
class PipelineConfig:
    """Полная конфигурация пайплайна отбора изображений."""
    # Пути
    input_directory: str
    output_json_path: str
    reference_directory: Optional[str] = None

    # Параметры отбора
    target_selection_size: int = 100
    max_clusters: int = MAX_CLUSTERS_DEFAULT
    batch_size: int = BATCH_SIZE_DEFAULT

    # Параметры MMR
    mmr_config: DiversityRelevanceConfig = field(default_factory=DiversityRelevanceConfig)

    # Параметры фильтрации
    outlier_contamination: float = 0.1  # Ожидаемая доля выбросов для Isolation Forest

    # Экспорт
    export_distance_matrix: bool = False

    # Модель
    clip_model_name: str = DEFAULT_CLIP_MODEL_NAME
    use_gpu: bool = True

    def __post_init__(self):
        if self.target_selection_size <= 0:
            raise ValueError("target_selection_size должен быть > 0")
        if not 0 < self.outlier_contamination <= 0.5:
            raise ValueError("outlier_contamination должен быть в диапазоне (0, 0.5]")


# =============================================================================
# ИНТЕРФЕЙСЫ (АБСТРАКЦИИ)
# =============================================================================

class ImageEmbeddingExtractor(ABC):
    """
    Интерфейс для извлечения векторных представлений (эмбеддингов) из изображений.

    Все реализации должны возвращать нормализованные эмбеддинги (L2-норма ≈ 1)
    для корректного вычисления косинусного сходства.
    """
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Размерность возвращаемых эмбеддингов."""
        ...

    @abstractmethod
    def extract_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """
        Извлекает эмбеддинги для пакета изображений.

        Args:
            images: Список RGB-изображений в формате numpy (HxWx3, uint8/float32)

        Returns:
            Матрица эмбеддингов формы (N, D), где N=len(images), D=embedding_dimension
        """
        ...

    @abstractmethod
    def extract_text(self, texts: list[str]) -> np.ndarray:
        """
        Извлекает эмбеддинги для текстовых запросов (для мультимодального поиска).

        Args:
            texts: Список текстовых строк

        Returns:
            Матрица эмбеддингов формы (N, D)
        """
        ...


class OutlierDetector(ABC):
    """Интерфейс детектора аномалий в пространстве эмбеддингов."""
    @abstractmethod
    def create_inlier_mask(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Определяет, какие образцы являются «нормальными» (не выбросами).

        Args:
            embeddings: Матрица эмбеддингов (N, D)

        Returns:
            Булев массив длины N: True = inlier, False = outlier
        """
        ...


class ClusterLabelAssigner(ABC):
    """Интерфейс для назначения кластерных меток эмбеддингам."""
    @abstractmethod
    def assign_cluster_labels(self, embeddings: np.ndarray, max_clusters: int) -> np.ndarray:
        """
        Назначает кластерные метки для группировки похожих изображений.

        Args:
            embeddings: Матрица эмбеддингов (N, D)
            max_clusters: Максимальное желаемое число кластеров

        Returns:
            Массив целочисленных меток кластеров длины N
        """
        ...


# =============================================================================
# РЕАЛИЗАЦИИ: CLIP Feature Extractor
# =============================================================================

class ClipEmbeddingExtractor(ImageEmbeddingExtractor):
    """
    Реализация извлечения эмбеддингов с использованием CLIP через sentence-transformers.

    Модель: sentence-transformers/clip-ViT-B-32 (или другая из библиотеки)
    - Изображения: 512-мерные эмбеддинги, L2-нормализованные
    - Тексты: совместимое пространство для кросс-модального поиска

    Требования:
        pip install sentence-transformers torch torchvision

    Примечания:
        - Автоматически определяет доступность CUDA
        - Кэширует модель после первой загрузки
        - Поддерживает пакетную обработку для эффективности
    """

    _model_cache: dict[str, SentenceTransformer] = {}  # Кэш моделей по имени

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL_NAME,
        use_gpu: bool = True,
        batch_size: int = BATCH_SIZE_DEFAULT,
    ):
        if SentenceTransformer is None:
            raise ImportError(
                "Требуется sentence-transformers. Установите: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.batch_size = batch_size

        # Инициализация модели с кэшированием
        if model_name not in self._model_cache:
            logger.info(f"Загрузка CLIP модели: {model_name}")
            device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            logger.info(f"Используется устройство: {device}")

            self._model_cache[model_name] = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True,
            )

        self._model = self._model_cache[model_name]
        self._embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"CLIP модель готова. Размерность эмбеддингов: {self._embedding_dim}")

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dim

    def extract_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Извлекает эмбеддинги для пакета изображений через PIL-конвертацию."""
        if not images:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        # Конвертация numpy → PIL для совместимости с sentence-transformers
        from PIL import Image
        pil_images = [
            Image.fromarray(_ensure_rgb(img)) if not isinstance(img, Image.Image) else img
            for img in images
        ]

        # Пакетное кодирование с прогресс-баром при необходимости
        embeddings = self._model.encode(
            pil_images,
            batch_size=self.batch_size,
            show_progress_bar=False,  # Управляем прогрессом на уровне пайплайна
            convert_to_numpy=True,
            normalize_embeddings=True,  # Важно: CLIP уже возвращает нормализованные вектора
        )
        return embeddings.astype(np.float32)

    def extract_text(self, texts: list[str]) -> np.ndarray:
        """Извлекает эмбеддинги для текстовых запросов."""
        if not texts:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


# =============================================================================
# РЕАЛИЗАЦИИ: Outlier Detector & Cluster Assigner
# =============================================================================

class IsolationForestOutlierDetector(OutlierDetector):
    """
    Детектор выбросов на основе Isolation Forest.

    Хорошо работает в высокоразмерных пространствах эмбеддингов.
    Параметр contamination задаёт ожидаемую долю аномалий.
    """
    def __init__(self, contamination: float = 0.1):
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination должен быть в диапазоне (0, 0.5]")
        self.contamination = contamination

    def create_inlier_mask(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) < 10:
            logger.warning("Слишком мало образцов для надёжного детектирования выбросов. Пропускаем.")
            return np.ones(len(embeddings), dtype=bool)

        model = IsolationForest(
            contamination=self.contamination,
            random_state=DEFAULT_RANDOM_STATE,
            n_jobs=-1,
            warm_start=False,
        )
        # fit_predict: 1 = inlier, -1 = outlier
        predictions = model.fit_predict(embeddings)
        return predictions == 1


class AdaptiveKMeansClusterAssigner(ClusterLabelAssigner):
    """
    K-Means кластеризация с адаптивным выбором числа кластеров.

    Эвристика: n_clusters = sqrt(n_samples / 2), ограничено [2, max_clusters]
    Обоснование: баланс между детализацией групп и статистической надёжностью.
    """
    def assign_cluster_labels(self, embeddings: np.ndarray, max_clusters: int) -> np.ndarray:
        n_samples = len(embeddings)

        if n_samples <= 2:
            return np.zeros(n_samples, dtype=int)

        # Эвристика выбора числа кластеров
        estimated_clusters = int(np.sqrt(n_samples / 2))
        n_clusters = max(2, min(max_clusters, estimated_clusters))

        logger.debug(f"Кластеризация: {n_samples} образцов → {n_clusters} кластеров")

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=DEFAULT_RANDOM_STATE,
            n_init=10,
            max_iter=300,
            n_jobs=-1,
        )
        return kmeans.fit_predict(embeddings)


# =============================================================================
# ЗАГРУЗКА И ЭКСПОРТ ДАННЫХ
# =============================================================================

def discover_image_paths(directory: str) -> list[Path]:
    """
    Рекурсивный поиск путей к изображениям в директории.

    Returns:
        Отсортированный список путей к валидным файлам изображений.
    """
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        raise ValueError(f"Директория не найдена: {dir_path}")

    paths = sorted(
        p for p in dir_path.rglob('*')
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and p.is_file()
    )
    logger.info(f"Найдено {len(paths)} изображений в {dir_path}")
    return paths


def load_image_batch(paths: list[Path]) -> tuple[list[np.ndarray], list[Path], int]:
    """
    Пакетная загрузка и предобработка изображений.

    Returns:
        images: Список изображений в формате RGB (numpy)
        valid_paths: Пути успешно загруженных файлов
        failed_count: Количество неудачных загрузок
    """
    images, valid_paths, failed = [], [], 0

    for p in paths:
        try:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                images.append(_ensure_rgb(img))
                valid_paths.append(p)
            else:
                failed += 1
                logger.debug(f"Пустое или битое изображение: {p}")
        except Exception as e:
            failed += 1
            logger.warning(f"Ошибка загрузки {p}: {e}")

    if failed:
        logger.warning(f"Пропущено {failed} из {len(paths)} изображений")

    return images, valid_paths, failed


@dataclass
class SelectionResult:
    """Контейнер результатов отбора с метаданными."""
    selected_paths: list[Path]
    relevance_scores: np.ndarray
    cluster_labels: np.ndarray
    original_embeddings: Optional[np.ndarray] = None  # Для дальнейшего анализа

    @property
    def count(self) -> int:
        return len(self.selected_paths)

    def to_dict_list(self) -> list[dict]:
        """Конвертация в список словарей для JSON-экспорта."""
        return [
            {
                "file_path": str(p),
                "relevance_score": float(score),
                "cluster_id": int(cluster),
            }
            for p, score, cluster in zip(
                self.selected_paths, self.relevance_scores, self.cluster_labels
            )
        ]


def export_selection_results_json(result: SelectionResult, output_path: str) -> None:
    """Экспорт результатов в структурированный JSON с метаданными."""
    if not result.selected_paths:
        logger.warning("Пустой результат — экспорт пропущен.")
        return

    # Статистика по релевантности
    rel_scores = result.relevance_scores
    relevance_stats = {
        "mean": float(np.mean(rel_scores)),
        "std": float(np.std(rel_scores)),
        "min": float(np.min(rel_scores)),
        "max": float(np.max(rel_scores)),
        "median": float(np.median(rel_scores)),
    }

    # Распределение по кластерам
    unique, counts = np.unique(result.cluster_labels, return_counts=True)
    cluster_distribution = {int(k): int(v) for k, v in zip(unique, counts)}

    payload = {
        "metadata": {
            "total_selected": result.count,
            "relevance_statistics": relevance_stats,
            "cluster_distribution": cluster_distribution,
            "embedding_dimension": result.original_embeddings.shape[1] if result.original_embeddings is not None else None,
        },
        "selection": result.to_dict_list(),
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Результаты сохранены: {out_file} ({result.count} изображений)")


def export_cosine_distance_matrix(embeddings: np.ndarray, base_output_path: str) -> None:
    """Экспорт матрицы попарных косинусных расстояний для оффлайн-анализа."""
    if len(embeddings) == 0:
        logger.warning("Пустые эмбеддинги — матрица расстояний не создана.")
        return

    normalized = _normalize_embeddings(embeddings)
    similarity_matrix = normalized @ normalized.T
    distance_matrix = 1.0 - _clip_similarity(similarity_matrix)
    np.fill_diagonal(distance_matrix, 0.0)

    dist_path = Path(base_output_path).with_suffix('.npy')
    np.save(dist_path, distance_matrix)
    logger.info(f"Матрица расстояний ({distance_matrix.shape}) сохранена: {dist_path}")


# =============================================================================
# ЯДРО: MMR-ОТБОР
# =============================================================================

def select_diverse_representatives_mmr(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    relevance_scores: np.ndarray,
    config: DiversityRelevanceConfig,
    target_size: int,
) -> list[int]:
    """
    Жадный отбор по принципу Maximal Marginal Relevance (MMR).

    Алгоритм:
    1. Гарантирует покрытие кластеров: выбирает наиболее релевантный образец из каждого
    2. Докручивает выборку, максимизируя маргинальную релевантность:
       score = λ*relevance + (1-λ)*(1 - max_similarity_to_selected)

    Сложность: O(k*n*d), где k=target_size, n=кандидаты, d=размерность

    Args:
        embeddings: Нормализованные эмбеддинги (N, D)
        cluster_labels: Метки кластеров для каждого образца
        relevance_scores: Предварительно вычисленные оценки релевантности [0, 1]
        config: Конфигурация баланса релевантность/разнообразие
        target_size: Желаемое количество отобранных образцов

    Returns:
        Список индексов отобранных образцов
    """
    n_samples = len(embeddings)

    # Edge cases
    if n_samples <= target_size:
        return list(range(n_samples))
    if n_samples == 0:
        return []

    # Предварительная нормализация (на всякий случай)
    normalized = _normalize_embeddings(embeddings)
    unique_clusters = np.unique(cluster_labels)

    # === ШАГ 1: Гарантированное покрытие кластеров ===
    cluster_representatives: list[int] = []

    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        indices_in_cluster = np.where(mask)[0]

        if len(indices_in_cluster) == 0:
            continue

        # Выбираем наиболее релевантный образец в кластере
        best_local_idx = np.argmax(relevance_scores[indices_in_cluster])
        global_idx = int(indices_in_cluster[best_local_idx])
        cluster_representatives.append(global_idx)

    # Сортируем представителей по релевантности (сначала лучшие)
    cluster_representatives.sort(key=lambda idx: relevance_scores[idx], reverse=True)

    # Инициализация выбранного набора
    selected: list[int] = []
    remaining = set(range(n_samples))

    for idx in cluster_representatives:
        if len(selected) >= target_size:
            break
        selected.append(idx)
        remaining.discard(idx)

    # === Инициализация максимального сходства с выбранными ===
    max_sim_to_selected = np.zeros(n_samples, dtype=np.float32)

    if selected:
        selected_embeds = normalized[selected]
        sim_to_selected = selected_embeds @ normalized.T  # (k, n)
        max_sim_to_selected = np.max(_clip_similarity(sim_to_selected), axis=0)

    # === ШАГ 2: Жадный добор по маргинальной релевантности ===
    while len(selected) < target_size and remaining:
        remaining_arr = np.fromiter(remaining, dtype=int, count=len(remaining))

        # Диверсификация: чем меньше схожесть с выбранными, тем лучше
        diversity_scores = 1.0 - max_sim_to_selected[remaining_arr]

        # Маргинальная релевантность: взвешенная сумма
        marginal_scores = (
            config.relevance_weight * relevance_scores[remaining_arr] +
            (1.0 - config.relevance_weight) * diversity_scores
        )

        # Выбор лучшего кандидата
        best_pos = int(np.argmax(marginal_scores))
        best_idx = int(remaining_arr[best_pos])

        # Обновление состояния
        selected.append(best_idx)
        remaining.remove(best_idx)

        # Эффективное обновление: только одна новая строка сходства
        new_sim = normalized @ normalized[best_idx]
        max_sim_to_selected = np.maximum(
            max_sim_to_selected,
            _clip_similarity(new_sim)
        )

    return selected


# =============================================================================
# ОРКЕСТРАТОР: ИЗОБРАЖЕНИЕ-ПАЙПЛАЙН
# =============================================================================

class ImageSelectionPipeline:
    """
    Полный пайплайн отбора изображений:
    загрузка → эмбеддинги → фильтрация → кластеризация → MMR-отбор → экспорт.

    Архитектура: композиция стратегий через dependency injection.
    """

    def __init__(
        self,
        extractor: ImageEmbeddingExtractor,
        outlier_detector: OutlierDetector,
        cluster_assigner: ClusterLabelAssigner,
        config: PipelineConfig,
    ):
        self.extractor = extractor
        self.outlier_detector = outlier_detector
        self.cluster_assigner = cluster_assigner
        self.config = config

        logger.info(
            f"Pipeline инициализирован: target={config.target_selection_size}, "
            f"MMR.λ={config.mmr_config.relevance_weight}, clusters≤{config.max_clusters}"
        )

    def _extract_embeddings_batched(self, paths: list[Path]) -> tuple[np.ndarray, list[Path]]:
        """Пакетное извлечение эмбеддингов с прогресс-индикацией."""
        if not paths:
            return np.empty((0, self.extractor.embedding_dimension)), []

        all_embeddings: list[np.ndarray] = []
        valid_paths: list[Path] = []

        progress_iter = tqdm(
            range(0, len(paths), self.config.batch_size),
            desc="Эмбеддинги",
            unit="батч"
        ) if tqdm else range(0, len(paths), self.config.batch_size)

        for start_idx in progress_iter:
            batch_paths = paths[start_idx : start_idx + self.config.batch_size]
            images, batch_valid, _ = load_image_batch(batch_paths)

            if images:
                batch_emb = self.extractor.extract_batch(images)
                all_embeddings.append(batch_emb)
                valid_paths.extend(batch_valid)

        if not all_embeddings:
            return np.empty((0, self.extractor.embedding_dimension)), []

        return np.vstack(all_embeddings), valid_paths

    def _calculate_relevance_metrics(
        self,
        embeddings: np.ndarray,
        reference_embeddings: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Вычисление релевантности: к референсам или к центроиду датасета.

        Возвращает нормализованные значения [0, 1].
        """
        if reference_embeddings is not None and len(reference_embeddings) > 0:
            # Кросс-модальное или внутри-модальное сходство с референсами
            raw_scores = np.max(
                cosine_similarity(embeddings, reference_embeddings),
                axis=1
            )
            logger.info(f"Релевантность вычислена относительно {len(reference_embeddings)} референсов")
        else:
            # Сходство с центроидом собственного датасета
            logger.info("Референсы не заданы. Используется центроид датасета.")
            centroid = embeddings.mean(axis=0, keepdims=True)
            raw_scores = cosine_similarity(embeddings, centroid).flatten()

        # Min-Max нормализация в [0, 1]
        min_score, max_score = raw_scores.min(), raw_scores.max()
        if max_score > min_score + EPSILON:
            return (raw_scores - min_score) / (max_score - min_score)
        return np.ones_like(raw_scores)

    def run(self) -> Optional[SelectionResult]:
        """
        Запуск полного пайплайна отбора.

        Returns:
            SelectionResult с отобранными путями и метаданными, или None при ошибке.
        """
        # === 1. Обнаружение изображений ===
        image_paths = discover_image_paths(self.config.input_directory)
        if not image_paths:
            logger.error(f"Нет изображений в {self.config.input_directory}")
            return None

        # === 2. Референсные эмбеддинги (опционально) ===
        reference_embeddings: Optional[np.ndarray] = None

        if self.config.reference_directory:
            ref_paths = discover_image_paths(self.config.reference_directory)
            if ref_paths:
                logger.info(f"Обработка {len(ref_paths)} референсных изображений...")
                reference_embeddings, _ = self._extract_embeddings_batched(ref_paths)
                logger.info(f"Референсные эмбеддинги: {reference_embeddings.shape}")

        # === 3. Эмбеддинги основного датасета ===
        logger.info(f"Обработка датасета: {len(image_paths)} изображений")
        dataset_embeddings, valid_paths = self._extract_embeddings_batched(image_paths)

        if len(valid_paths) == 0:
            logger.error("Не удалось загрузить ни одного изображения.")
            return None

        logger.info(f"Эмбеддинги датасета: {dataset_embeddings.shape}")

        # === 4. Фильтрация выбросов ===
        if len(valid_paths) > self.config.target_selection_size:
            logger.info("Фильтрация выбросов (Isolation Forest)...")
            inlier_mask = self.outlier_detector.create_inlier_mask(dataset_embeddings)

            n_removed = np.sum(~inlier_mask)
            if n_removed > 0:
                logger.info(f"Удалено {n_removed} выбросов ({100*n_removed/len(inlier_mask):.1f}%)")

            dataset_embeddings = dataset_embeddings[inlier_mask]
            valid_paths = [p for p, keep in zip(valid_paths, inlier_mask) if keep]

        # === 5. Кластеризация ===
        cluster_labels = np.zeros(len(valid_paths), dtype=int)

        if len(valid_paths) > self.config.target_selection_size:
            logger.info("Кластеризация (Adaptive K-Means)...")
            cluster_labels = self.cluster_assigner.assign_cluster_labels(
                dataset_embeddings,
                max_clusters=self.config.max_clusters,
            )
        else:
            logger.info(
                f"Пропуск кластеризации: {len(valid_paths)} образцов ≤ "
                f"target {self.config.target_selection_size}"
            )

        # === 6. Расчёт релевантности ===
        logger.info("Расчёт метрик релевантности...")
        relevance_scores = self._calculate_relevance_metrics(
            dataset_embeddings, reference_embeddings
        )

        # === 7. MMR-отбор (если нужно сократить) ===
        if len(valid_paths) > self.config.target_selection_size:
            logger.info(
                f"MMR-отбор: {len(valid_paths)} → {self.config.target_selection_selection_size}"
            )
            selected_indices = select_diverse_representatives_mmr(
                embeddings=dataset_embeddings,
                cluster_labels=cluster_labels,
                relevance_scores=relevance_scores,
                config=self.config.mmr_config,
                target_size=self.config.target_selection_size,
            )

            # Применение маски отбора
            dataset_embeddings = dataset_embeddings[selected_indices]
            valid_paths = [valid_paths[i] for i in selected_indices]
            relevance_scores = relevance_scores[selected_indices]
            cluster_labels = cluster_labels[selected_indices]
        else:
            logger.info(
                f"Пропуск MMR: {len(valid_paths)} образцов ≤ target {self.config.target_selection_size}"
            )

        # === 8. Экспорт результатов ===
        if self.config.export_distance_matrix:
            export_cosine_distance_matrix(dataset_embeddings, self.config.output_json_path)

        result = SelectionResult(
            selected_paths=valid_paths,
            relevance_scores=relevance_scores,
            cluster_labels=cluster_labels,
            original_embeddings=dataset_embeddings,
        )

        export_selection_results_json(result, self.config.output_json_path)

        logger.info(f"✅ Пайплайн завершён. Отобрано {result.count} изображений.")
        return result


# =============================================================================
# CLI: КОНСОЛЬНЫЙ ИНТЕРФЕЙС
# =============================================================================

def _parse_arguments() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="🎨 Image Selection Pipeline — интеллектуальный отбор датасета для LoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Примеры:
  # Базовый запуск
  python selector.py --input-dir ./photos --target-size 200
  
  # С референсами стиля и акцентом на разнообразие
  python selector.py --input-dir ./raw --reference-dir ./style_ref \\
                     --target-size 150 --relevance-weight 0.3
  
  # Сохранение матрицы расстояний для анализа
  python selector.py --input-dir ./data --target-size 100 \\
                     --save-dist-matrix --output results/selection.json
        """,
    )

    # Обязательные аргументы
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Директория с исходными изображениями (рекурсивный поиск)"
    )
    parser.add_argument(
        "--target-size", type=int, required=True,
        help="Желаемое количество изображений в финальной выборке"
    )

    # Опциональные пути
    parser.add_argument(
        "--output-json", type=str, default="selection_result.json",
        help="Путь к выходному JSON-файлу"
    )
    parser.add_argument(
        "--reference-dir", type=str, default=None,
        help="Директория с референсными изображениями (стиль/тема)"
    )

    # Параметры MMR и кластеризации
    parser.add_argument(
        "--relevance-weight", type=float, default=0.5,
        help="Вес релевантности в MMR: 1.0=только релевантность, 0.0=только разнообразие"
    )
    parser.add_argument(
        "--max-clusters", type=int, default=MAX_CLUSTERS_DEFAULT,
        help="Максимальное число кластеров для K-Means"
    )

    # Параметры фильтрации и модели
    parser.add_argument(
        "--contamination", type=float, default=0.1,
        help="Ожидаемая доля выбросов для Isolation Forest"
    )
    parser.add_argument(
        "--clip-model", type=str, default=DEFAULT_CLIP_MODEL_NAME,
        help="Имя CLIP-модели из sentence-transformers"
    )
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Принудительно использовать CPU (игнорировать CUDA)"
    )

    # Технические параметры
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE_DEFAULT,
        help="Размер батча для извлечения эмбеддингов"
    )
    parser.add_argument(
        "--save-dist-matrix", action="store_true",
        help="Сохранить матрицу косинусных расстояний (.npy) для анализа"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Включить отладочное логирование"
    )

    return parser.parse_args()


def main() -> int:
    """Точка входа CLI."""
    args = _parse_arguments()

    # Настройка логирования
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Валидация входных данных
    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        logger.error(f"❌ Директория не найдена: {input_path}")
        return 1

    # Сборка конфигурации
    config = PipelineConfig(
        input_directory=args.input_dir,
        output_json_path=args.output_json,
        reference_directory=args.reference_dir,
        target_selection_size=args.target_size,
        max_clusters=args.max_clusters,
        batch_size=args.batch_size,
        mmr_config=DiversityRelevanceConfig(
            relevance_weight=args.relevance_weight
        ),
        outlier_contamination=args.contamination,
        export_distance_matrix=args.save_dist_matrix,
        clip_model_name=args.clip_model,
        use_gpu=not args.cpu_only,
    )

    # Инициализация компонентов
    try:
        extractor = ClipEmbeddingExtractor(
            model_name=config.clip_model_name,
            use_gpu=config.use_gpu,
            batch_size=config.batch_size,
        )
    except ImportError as e:
        logger.error(f"❌ Ошибка инициализации CLIP: {e}")
        logger.info("💡 Установите: pip install sentence-transformers torch torchvision")
        return 1

    outlier_detector = IsolationForestOutlierDetector(
        contamination=config.outlier_contamination
    )

    cluster_assigner = AdaptiveKMeansClusterAssigner()

    # Запуск пайплайна
    pipeline = ImageSelectionPipeline(
        extractor=extractor,
        outlier_detector=outlier_detector,
        cluster_assigner=cluster_assigner,
        config=config,
    )

    result = pipeline.run()

    return 0 if result is not None else 1


if __name__ == "__main__":
    # Lazy import torch только при необходимости
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore

    exit_code = main()
    exit(exit_code)