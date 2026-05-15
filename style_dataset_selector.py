"""
Продвинутая ООП Система отбора датасета для LoRA (улучшенная версия)
====================================================================
Запуск:
python script.py --input-dir ./my_images --target-size 30 --output-json result.json
(Опционально) --reference-dir ./style_references --save-dist-matrix

Изменения относительно базовой версии:
- Нормализация штрафа за разнообразие (MMR) для единого диапазона с релевантностью.
- Более безопасная заглушка tqdm.
- Упрощённая инициализация меток кластеров.
- Говорящие имена классов и переменных.
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
    raise ImportError(
        f"Требуется scikit-learn. Установите: pip install scikit-learn\n{e}"
    ) from e

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        """Безопасная заглушка tqdm, возвращающая iterable без изменений."""
        return iterable if iterable is not None else []

logger = logging.getLogger(__name__)


# =============================================================================
# 1. КОНФИГУРАЦИЯ И ИНТЕРФЕЙСЫ
# =============================================================================

@dataclass
class MmrBalanceWeights:
    """Веса для алгоритма MMR: Relevance (сходство) vs Diversity (разнообразие)."""
    relevance_weight: float = 0.5
    diversity_weight: float = 0.5

    def __post_init__(self):
        if self.relevance_weight < 0 or self.diversity_weight < 0:
            raise ValueError("Веса должны быть неотрицательными.")
        total = self.relevance_weight + self.diversity_weight
        if total <= 0:
            raise ValueError("Сумма весов должна быть больше 0.")
        # Нормализация
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

class RandomEmbeddingModel(EmbeddingModel):
    """Генерирует воспроизводимые случайные эмбеддинги (заглушка).
    В продакшене заменить на CLIP, DINOv2, EVA и т.д.
    """
    def __init__(self, dim: int = 512, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._dim = dim
        logger.warning("RandomEmbeddingModel активна: эмбеддинги генерируются случайно.")

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        return self._rng.standard_normal((len(images), self._dim)).astype(np.float32)


class IsolationForestDetector(OutlierDetector):
    def __init__(self, contamination: float = 0.1):
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination должен быть в диапазоне (0, 0.5]")
        self.contamination = contamination

    def detect(self, embeddings: np.ndarray) -> np.ndarray:
        model = IsolationForest(
            contamination=self.contamination, random_state=42, n_jobs=-1
        )
        return model.fit_predict(embeddings) != -1


class KMeansImageClusterer(Clusterer):
    def __init__(self, max_clusters: int = 20):
        if max_clusters < 2:
            raise ValueError("max_clusters должен быть >= 2")
        self.max_clusters = max_clusters

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        n = len(embeddings)
        # Для маленьких датасетов нужно хотя бы 2 кластера
        n_clusters = max(2, min(self.max_clusters, int(np.sqrt(n / 2))))
        return KMeans(
            n_clusters=n_clusters, random_state=42, n_init='auto'
        ).fit_predict(embeddings)


# =============================================================================
# 3. ЗАГРУЗЧИК И ЭКСПОРТЕР
# =============================================================================

class ImageDirectoryLoader:
    """Поиск путей и пакетная загрузка изображений из директории."""
    SUPPORTED_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.bmp'})

    def __init__(self, directory: str):
        self.directory = Path(directory).resolve()
        if not self.directory.is_dir():
            raise ValueError(f"Директория не найдена: {self.directory}")

    def get_paths(self) -> list[Path]:
        paths = sorted(
            p for p in self.directory.rglob('*')
            if p.suffix.lower() in self.SUPPORTED_EXTENSIONS and p.is_file()
        )
        logger.info(f"Найдено {len(paths)} изображений в {self.directory}")
        return paths

    @staticmethod
    def load_batch(paths: list[Path]) -> tuple[list[np.ndarray], list[Path]]:
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
class CurationResult:
    """Контейнер результата курирования для передачи в экспортер."""
    paths: list[Path]
    embeddings: np.ndarray
    scores: np.ndarray
    clusters: np.ndarray


class SelectionJsonExporter:
    @staticmethod
    def export(result: CurationResult, output_path: str,
               save_dist_matrix: bool = False) -> None:
        if not result.paths:
            logger.warning("Нечего экспортировать — пустая выборка.")
            return

        selection_data = [
            {
                "file_path": str(p),
                "relevance_score": float(score),
                "cluster_id": int(cluster),
            }
            for p, score, cluster in zip(result.paths, result.scores, result.clusters)
        ]

        payload = {
            "metadata": {
                "total_selected": len(result.paths),
                "avg_relevance": float(np.mean(result.scores)),
            },
            "selection": selection_data,
        }

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
        logger.info(f"Результат сохранён в {out_file}")

        if save_dist_matrix:
            # Косинусное расстояние: 1 - сходство (используем уже нормализованные эмбеддинги)
            normed = result.embeddings / (
                np.linalg.norm(result.embeddings, axis=1, keepdims=True) + 1e-10
            )
            dist_matrix = 1.0 - (normed @ normed.T)
            np.fill_diagonal(dist_matrix, 0.0)
            dist_path = out_file.with_suffix('.npy')
            np.save(dist_path, dist_matrix)
            logger.info(f"Матрица расстояний сохранена в {dist_path}")


# =============================================================================
# 4. ЯДРО ЛОГИКИ ОТБОРА (MMR)
# =============================================================================

class MaximalMarginalRelevanceSelector:
    """Maximal Marginal Relevance: баланс репрезентативности и разнообразия."""

    @staticmethod
    def select(
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        relevance_scores: np.ndarray,
        weights: MmrBalanceWeights,
        target_size: int,
    ) -> list[int]:
        n = len(embeddings)
        if n <= target_size:
            return list(range(n))

        normalized_embeddings = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        )
        unique_labels = np.unique(cluster_labels)

        # 1. Гарантируем покрытие кластеров
        cluster_representatives: list[int] = []
        for label in unique_labels:
            indices_in_cluster = np.where(cluster_labels == label)[0]
            best_index_in_cluster = indices_in_cluster[np.argmax(relevance_scores[indices_in_cluster])]
            cluster_representatives.append(best_index_in_cluster)

        # Сортируем представителей по релевантности
        cluster_representatives.sort(key=lambda idx: relevance_scores[idx], reverse=True)

        selected: list[int] = []
        remaining = set(range(n))

        # Добавляем представителей кластеров (не больше, чем target_size)
        for rep_idx in cluster_representatives:
            if len(selected) >= target_size:
                break
            selected.append(rep_idx)
            remaining.discard(rep_idx)

        # 2. Жадный MMR добор
        while len(selected) < target_size and remaining:
            remaining_indices = np.array(list(remaining))

            # Сходство кандидатов с уже выбранными
            sim_to_selected = normalized_embeddings[remaining_indices] @ normalized_embeddings[selected].T

            # Максимальное сходство с любым уже выбранным элементом
            max_sim = np.max(sim_to_selected, axis=1)

            # Diversity: штрафуем за похожесть (приводим к диапазону ~[0,1], как у relevance)
            diversity = 0.5 * np.maximum(0.0, 1.0 - max_sim)

            mmr_scores = (
                weights.relevance_weight * relevance_scores[remaining_indices]
                + weights.diversity_weight * diversity
            )

            best_in_remaining_idx = np.argmax(mmr_scores)
            best_overall_idx = int(remaining_indices[best_in_remaining_idx])

            selected.append(best_overall_idx)
            remaining.remove(best_overall_idx)

        return selected


# =============================================================================
# 5. ОРКЕСТРАТОР ПАЙПЛАЙНА
# =============================================================================

class DatasetCurationPipeline:
    def __init__(
        self,
        model: EmbeddingModel,
        detector: OutlierDetector,
        clusterer: Clusterer,
        weights: MmrBalanceWeights,
        target_size: int,
        batch_size: int = 32,
    ):
        if target_size <= 0:
            raise ValueError("target_size должен быть > 0")
        self.model = model
        self.detector = detector
        self.clusterer = clusterer
        self.weights = weights
        self.target_size = target_size
        self.batch_size = batch_size

    def _compute_embeddings(self, paths: list[Path]) -> tuple[np.ndarray, list[Path]]:
        all_emb: list[np.ndarray] = []
        valid_paths: list[Path] = []

        for i in tqdm(range(0, len(paths), self.batch_size), desc="Кодирование изображений"):
            batch_paths = paths[i : i + self.batch_size]
            images, batch_valid = ImageDirectoryLoader.load_batch(batch_paths)
            if images:
                all_emb.append(self.model.encode(images))
                valid_paths.extend(batch_valid)

        if not all_emb:
            return np.empty((0, 0)), []
        return np.vstack(all_emb), valid_paths

    @staticmethod
    def _compute_relevance_scores(
        embeddings: np.ndarray,
        reference_embeddings: np.ndarray | None,
    ) -> np.ndarray:
        """Сходство с центром референсов. Если референсов нет – репрезентативность относительно центра датасета."""
        if reference_embeddings is not None and len(reference_embeddings) > 0:
            center = reference_embeddings.mean(axis=0, keepdims=True)
        else:
            logger.info("Референсы не заданы. Скоринг относительно центра датасета (репрезентативность).")
            center = embeddings.mean(axis=0, keepdims=True)
        return cosine_similarity(embeddings, center).flatten()

    def process(
        self,
        input_dir: str,
        reference_dir: str | None,
        output_json: str,
        save_dist_matrix: bool = False,
    ) -> None:
        loader = ImageDirectoryLoader(input_dir)
        image_paths = loader.get_paths()

        if not image_paths:
            logger.warning("Нет изображений для обработки.")
            return

        # 1. Референсы
        reference_embeddings: np.ndarray | None = None
        if reference_dir:
            ref_loader = ImageDirectoryLoader(reference_dir)
            ref_paths = ref_loader.get_paths()
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

        # 3. Фильтрация выбросов и кластеризация (только если данных больше целевого)
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

        # 5. MMR Отбор (если нужно)
        if len(current_paths) > self.target_size:
            logger.info(f"Отбор {self.target_size} изображений по MMR...")
            selected_indices = MaximalMarginalRelevanceSelector.select(
                embeddings=current_embeddings,
                cluster_labels=current_clusters,
                relevance_scores=relevance_scores,
                weights=self.weights,
                target_size=self.target_size,
            )
            current_embeddings = current_embeddings[selected_indices]
            current_paths = [current_paths[i] for i in selected_indices]
            relevance_scores = relevance_scores[selected_indices]
            current_clusters = current_clusters[selected_indices]

        # 6. Экспорт
        result = CurationResult(
            paths=current_paths,
            embeddings=current_embeddings,
            scores=relevance_scores,
            clusters=current_clusters,
        )
        SelectionJsonExporter.export(result, output_json, save_dist_matrix)


# =============================================================================
# 6. КОНСОЛЬНЫЙ ИНТЕРФЕЙС
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ООП Система отбора и курирования датасета для LoRA (улучшенная версия)."
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
    model = RandomEmbeddingModel()
    detector = IsolationForestDetector(contamination=args.contamination)
    clusterer = KMeansImageClusterer(max_clusters=args.max_clusters)
    weights = MmrBalanceWeights(
        relevance_weight=args.relevance_weight,
        diversity_weight=args.diversity_weight,
    )

    pipeline = DatasetCurationPipeline(
        model=model,
        detector=detector,
        clusterer=clusterer,
        weights=weights,
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