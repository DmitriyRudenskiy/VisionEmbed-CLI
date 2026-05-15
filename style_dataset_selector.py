"""
Продвинутая ООП Система отбора датасета для LoRA (v2 - Refactored)
====================================================================
Запуск:
python script.py --input-dir ./my_images --target-size 30 --output-json result.json
(Опционально) --reference-dir ./style_references --save-dist-matrix

Ключевые улучшения:
- Честный баланс весов в MMR (убран скрытый коэффициент 0.5).
- Robust скоринг относительно референсов (Max Similarity вместо Mean).
- Нормализация векторов в заглушке (для корректной работы косинусного сходства).
- Вынос загрузки батча и сохранения матрицы из классов-экспортеров.
- Говорящие имена переменных и классов.
"""

import json
import argparse
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# -----------------------------------------------------------------------------
# Зависимости
# -----------------------------------------------------------------------------
try:
    import cv2
except ImportError as e:
    raise ImportError("Требуется OpenCV. Установите: pip install opencv-python") from e

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
except ImportError as e:
    raise ImportError(f"Требуется scikit-learn. Установите: pip install scikit-learn\n{e}") from e

try:
    from tqdm import tqdm
except ImportError:
    # Безопасная заглушка, сохраняющая интерфейс tqdm (поддержка desc, total и т.д.)
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

logger = logging.getLogger(__name__)


# =============================================================================
# 1. КОНФИГУРАЦИЯ И ИНТЕРФЕЙСЫ
# =============================================================================

@dataclass
class MmrConfig:
    """Конфигурация весов MMR: Relevance (сходство) vs Diversity (разнообразие)."""
    relevance_weight: float = 0.5
    diversity_weight: float = 0.5

    def __post_init__(self):
        if self.relevance_weight < 0 or self.diversity_weight < 0:
            raise ValueError("Веса должны быть неотрицательными.")
        if self.relevance_weight + self.diversity_weight == 0:
            raise ValueError("Сумма весов должна быть больше 0.")
        # Нормализация к 1.0
        total = self.relevance_weight + self.diversity_weight
        self.relevance_weight /= total
        self.diversity_weight /= total


class EmbeddingModel(ABC):
    """Интерфейс модели эмбеддингов."""
    @abstractmethod
    def encode(self, images: list[np.ndarray]) -> np.ndarray: ...


class OutlierDetector(ABC):
    @abstractmethod
    def detect(self, embeddings: np.ndarray) -> np.ndarray:
        """Возвращает булеву маску (True — валидный, False — выброс)."""
        ...


class Clusterer(ABC):
    @abstractmethod
    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Возвращает массив меток кластеров."""
        ...


# =============================================================================
# 2. КОНКРЕТНЫЕ РЕАЛИЗАЦИИ СТРАТЕГИЙ
# =============================================================================

class StubEmbeddingModel(EmbeddingModel):
    """Генерирует воспроизводимые случайные эмбеддинги, нормализованные к единичной сфере (заглушка)."""
    def __init__(self, dim: int = 512, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._dim = dim
        logger.warning("StubEmbeddingModel активна: эмбеддинги генерируются случайно.")

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        raw_emb = self._rng.standard_normal((len(images), self._dim)).astype(np.float32)
        # Нормализация обязательна для корректной работы косинусного сходства в MMR
        norms = np.linalg.norm(raw_emb, axis=1, keepdims=True) + 1e-10
        return raw_emb / norms


class IsolationForestDetector(OutlierDetector):
    def __init__(self, contamination: float = 0.1):
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination должен быть в диапазоне (0, 0.5]")
        self.contamination = contamination

    def detect(self, embeddings: np.ndarray) -> np.ndarray:
        # Внимание: IsolationForest плохо работает в высокоразмерных пространствах (>100 dim).
        # Для продакшена рекомендуется предварительно применять PCA или UMAP.
        model = IsolationForest(
            contamination=self.contamination, random_state=42, n_jobs=-1
        )
        return model.fit_predict(embeddings) != -1


class KMeansClusterer(Clusterer):
    def __init__(self, max_clusters: int = 20):
        if max_clusters < 2:
            raise ValueError("max_clusters должен быть >= 2")
        self.max_clusters = max_clusters

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        n_samples = len(embeddings)
        n_clusters = max(2, min(self.max_clusters, int(np.sqrt(n_samples / 2))))
        return KMeans(
            n_clusters=n_clusters, random_state=42, n_init='auto'
        ).fit_predict(embeddings)


# =============================================================================
# 3. УТИЛИТЫ ЗАГРУЗКИ И ЭКСПОРТА
# =============================================================================

SUPPORTED_IMAGE_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.bmp'})

def scan_image_directory(directory: str) -> list[Path]:
    """Рекурсивный поиск путей изображений в директории."""
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        raise ValueError(f"Директория не найдена: {dir_path}")

    paths = sorted(
        p for p in dir_path.rglob('*')
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and p.is_file()
    )
    logger.info(f"Найдено {len(paths)} изображений в {dir_path}")
    return paths


def load_image_batch(paths: list[Path]) -> tuple[list[np.ndarray], list[Path]]:
    """Пакетная загрузка и конвертация изображений (BGR -> RGB)."""
    images, valid_paths = [], []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            valid_paths.append(p)
        else:
            logger.warning(f"Не удалось загрузить (битый файл?): {p}")
    return images, valid_paths


@dataclass
class DatasetSelection:
    """Контейнер результата отбора."""
    paths: list[Path]
    embeddings: np.ndarray
    relevance_scores: np.ndarray
    cluster_labels: np.ndarray


class JsonExporter:
    @staticmethod
    def export(selection: DatasetSelection, output_path: str) -> None:
        if not selection.paths:
            logger.warning("Нечего экспортировать — пустая выборка.")
            return

        selection_data = [
            {
                "file_path": str(p),
                "relevance_score": float(score),
                "cluster_id": int(cluster),
            }
            for p, score, cluster in zip(
                selection.paths, selection.relevance_scores, selection.cluster_labels
            )
        ]

        payload = {
            "metadata": {
                "total_selected": len(selection.paths),
                "avg_relevance": float(np.mean(selection.relevance_scores)),
            },
            "selection": selection_data,
        }

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        logger.info(f"Результат сохранён в {out_file}")


# =============================================================================
# 4. ЯДРО ЛОГИКИ ОТБОРА (MMR)
# =============================================================================

class MmrSelector:
    """Maximal Marginal Relevance: баланс репрезентативности и разнообразия."""

    @staticmethod
    def select(
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        relevance_scores: np.ndarray,
        config: MmrConfig,
        target_size: int,
    ) -> list[int]:
        n_samples = len(embeddings)
        if n_samples <= target_size:
            return list(range(n_samples))

        # Эмбеддинги уже нормализованы на этапе генерации/извлечения, но для надежности убедимся
        normed_embeddings = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        )

        unique_labels = np.unique(cluster_labels)

        # 1. Гарантируем покрытие кластеров (выбираем лучших представителей)
        cluster_representatives: list[int] = []
        for label in unique_labels:
            indices_in_cluster = np.where(cluster_labels == label)[0]
            best_index = indices_in_cluster[np.argmax(relevance_scores[indices_in_cluster])]
            cluster_representatives.append(best_index)

        # Сортируем представителей по релевантности
        cluster_representatives.sort(key=lambda idx: relevance_scores[idx], reverse=True)

        selected_indices: list[int] = []
        remaining_indices = set(range(n_samples))

        # Добавляем представителей кластеров (не больше, чем target_size)
        for rep_idx in cluster_representatives:
            if len(selected_indices) >= target_size:
                break
            selected_indices.append(rep_idx)
            remaining_indices.discard(rep_idx)

        # 2. Жадный MMR добор
        while len(selected_indices) < target_size and remaining_indices:
            remaining_list = np.array(list(remaining_indices))

            # Сходство кандидатов с уже выбранными
            similarity_to_selected = normed_embeddings[remaining_list] @ normed_embeddings[selected_indices].T

            # Максимальное сходство кандидата с любым уже выбранным элементом
            max_similarity = np.max(similarity_to_selected, axis=1)

            # Diversity: штрафуем за похожесть. (1 - max_sim) уже дает диапазон [0, 1], как у relevance
            diversity = 1.0 - max_similarity

            # Честный взвешенный скоринг (без скрытых множителей)
            mmr_scores = (
                config.relevance_weight * relevance_scores[remaining_list]
                + config.diversity_weight * diversity
            )

            best_candidate_pos = np.argmax(mmr_scores)
            best_candidate_idx = int(remaining_list[best_candidate_pos])

            selected_indices.append(best_candidate_idx)
            remaining_indices.remove(best_candidate_idx)

        return selected_indices


# =============================================================================
# 5. ОРКЕСТРАТОР ПАЙПЛАЙНА
# =============================================================================

class DatasetCurationPipeline:
    def __init__(
        self,
        model: EmbeddingModel,
        detector: OutlierDetector,
        clusterer: Clusterer,
        mmr_config: MmrConfig,
        target_size: int,
        batch_size: int = 32,
    ):
        if target_size <= 0:
            raise ValueError("target_size должен быть > 0")
        self.model = model
        self.detector = detector
        self.clusterer = clusterer
        self.mmr_config = mmr_config
        self.target_size = target_size
        self.batch_size = batch_size

    def _compute_embeddings(self, paths: list[Path]) -> tuple[np.ndarray, list[Path]]:
        all_embeddings: list[np.ndarray] = []
        valid_paths: list[Path] = []

        for i in tqdm(range(0, len(paths), self.batch_size), desc="Кодирование изображений"):
            batch_paths = paths[i : i + self.batch_size]
            images, batch_valid = load_image_batch(batch_paths)
            if images:
                all_embeddings.append(self.model.encode(images))
                valid_paths.extend(batch_valid)

        if not all_embeddings:
            return np.empty((0, 0)), []
        return np.vstack(all_embeddings), valid_paths

    @staticmethod
    def _compute_relevance_scores(
        embeddings: np.ndarray,
        reference_embeddings: np.ndarray | None,
    ) -> np.ndarray:
        """Сходство с референсами. Если референсов нет – репрезентативность относительно центра датасета."""
        if reference_embeddings is not None and len(reference_embeddings) > 0:
            # Max Similarity: изображение считается релевантным, если похоже хотя бы на один референс.
            # Это робастнее, чем усреднение (Mean), которое "размывает" стиль.
            return np.max(cosine_similarity(embeddings, reference_embeddings), axis=1)
        else:
            logger.info("Референсы не заданы. Скоринг относительно центра датасета (репрезентативность).")
            center = embeddings.mean(axis=0, keepdims=True)
            return cosine_similarity(embeddings, center).flatten()

    @staticmethod
    def save_distance_matrix(embeddings: np.ndarray, output_path: str) -> None:
        """Сохраняет матрицу косинусных расстояний (.npy)."""
        normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        dist_matrix = 1.0 - (normed @ normed.T)
        np.fill_diagonal(dist_matrix, 0.0)
        dist_path = Path(output_path).with_suffix('.npy')
        np.save(dist_path, dist_matrix)
        logger.info(f"Матрица расстояний сохранена в {dist_path}")

    def process(
        self,
        input_dir: str,
        reference_dir: str | None,
        output_json: str,
        save_dist_matrix: bool = False,
    ) -> None:
        image_paths = scan_image_directory(input_dir)

        if not image_paths:
            logger.warning("Нет изображений для обработки.")
            return

        # 1. Референсы
        reference_embeddings: np.ndarray | None = None
        if reference_dir:
            ref_paths = scan_image_directory(reference_dir)
            if ref_paths:
                logger.info(f"Вычисление эмбеддингов для {len(ref_paths)} референсов...")
                reference_embeddings, _ = self._compute_embeddings(ref_paths)

        # 2. Основной датасет
        logger.info("Вычисление эмбеддингов датасета...")
        current_embeddings, current_paths = self._compute_embeddings(image_paths)

        if not current_paths:
            logger.warning("Все изображения оказались битыми или не поддерживаются.")
            return

        # Инициализация кластерных меток (по умолчанию всё в одном кластере)
        current_clusters = np.zeros(len(current_paths), dtype=int)

        # 3. Фильтрация выбросов и кластеризация
        if len(current_paths) > self.target_size:
            logger.info("Фильтрация выбросов...")
            valid_samples_mask = self.detector.detect(current_embeddings)
            current_embeddings = current_embeddings[valid_samples_mask]
            current_paths = [p for p, is_valid in zip(current_paths, valid_samples_mask) if is_valid]

            if len(current_paths) > self.target_size:
                logger.info("Кластеризация...")
                current_clusters = self.clusterer.cluster(current_embeddings)
            else:
                logger.info(f"После фильтрации осталось {len(current_paths)} ≤ target_size. Кластеризация пропущена.")
        else:
            logger.info("Размер датасета ≤ target_size, MMR и фильтрация не требуются.")

        # 4. Скоринг
        logger.info("Вычисление метрик релевантности...")
        relevance_scores = self._compute_relevance_scores(current_embeddings, reference_embeddings)

        # 5. MMR Отбор
        if len(current_paths) > self.target_size:
            logger.info(f"Отбор {self.target_size} изображений по MMR...")
            selected_indices = MmrSelector.select(
                embeddings=current_embeddings,
                cluster_labels=current_clusters,
                relevance_scores=relevance_scores,
                config=self.mmr_config,
                target_size=self.target_size,
            )
            current_embeddings = current_embeddings[selected_indices]
            current_paths = [current_paths[i] for i in selected_indices]
            relevance_scores = relevance_scores[selected_indices]
            current_clusters = current_clusters[selected_indices]

        # 6. Экспорт
        result = DatasetSelection(
            paths=current_paths,
            embeddings=current_embeddings,
            relevance_scores=relevance_scores,
            cluster_labels=current_clusters,
        )
        JsonExporter.export(result, output_json)

        if save_dist_matrix:
            self.save_distance_matrix(result.embeddings, output_json)


# =============================================================================
# 6. КОНСОЛЬНЫЙ ИНТЕРФЕЙС
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ООП Система отбора и курирования датасета для LoRA (v2)."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Папка с исходными картинками")
    parser.add_argument("--target-size", type=int, required=True, help="Желаемое количество картинок в выборке")
    parser.add_argument("--output-json", type=str, default="result.json", help="Путь к JSON файлу")
    parser.add_argument("--reference-dir", type=str, default=None, help="Папка с эталонными картинками стиля")
    parser.add_argument("--save-dist-matrix", action="store_true", help="Сохранить матрицу расстояний (.npy)")
    parser.add_argument("--contamination", type=float, default=0.1, help="Доля выбросов для IsolationForest")
    parser.add_argument("--max-clusters", type=int, default=20, help="Макс. число кластеров для KMeans")
    parser.add_argument("--relevance-weight", type=float, default=0.5, help="Вес схожести/репрезентативности (MMR)")
    parser.add_argument("--diversity-weight", type=float, default=0.5, help="Вес разнообразия (MMR)")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча при кодировании")
    args = parser.parse_args()

    if not Path(args.input_dir).is_dir():
        logger.error(f"Директория '{args.input_dir}' не найдена.")
        return

    # Инициализация компонентов
    model = StubEmbeddingModel()
    detector = IsolationForestDetector(contamination=args.contamination)
    clusterer = KMeansClusterer(max_clusters=args.max_clusters)
    mmr_config = MmrConfig(
        relevance_weight=args.relevance_weight,
        diversity_weight=args.diversity_weight,
    )

    pipeline = DatasetCurationPipeline(
        model=model,
        detector=detector,
        clusterer=clusterer,
        mmr_config=mmr_config,
        target_size=args.target_size,
        batch_size=args.batch_size,
    )

    pipeline.process(
        input_dir=args.input_dir,
        reference_dir=args.reference_dir,
        output_json=args.output_json,
        save_dist_matrix=args.save_dist_matrix,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()