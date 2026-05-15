"""
Продвинутая ООП Система отбора датасета для LoRA
=================================================
Запуск:
python script.py --input-dir ./my_images --target-size 30 --output-json result.json
(Опционально) --reference-dir ./style_references
"""

import os
import glob
import json
import argparse
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Импорты зависимостей
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.cluster import KMeans, DBSCAN
except ImportError as e:
    raise ImportError(f"Требуется scikit-learn. Установите: pip install scikit-learn\n{e}")

logger = logging.getLogger(__name__)


# =============================================================================
# 1. КОНФИГУРАЦИЯ И ИНТЕРФЕЙСЫ (STRATEGY PATTERN)
# =============================================================================

@dataclass
class SelectionCriteria:
    similarity_threshold: float = 0.5
    similarity_weight: float = 0.5
    diversity_weight: float = 0.5

class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, images: List[np.ndarray], batch_size: int = 32) -> np.ndarray: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

class OutlierDetector(ABC):
    @abstractmethod
    def detect(self, embeddings: np.ndarray) -> np.ndarray:
        """Возвращает булеву маску (True - валидный, False - выброс)"""
        ...

class Clusterer(ABC):
    @abstractmethod
    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Возвращает массив меток кластеров"""
        ...


# =============================================================================
# 2. КОНКРЕТНЫЕ РЕАЛИЗАЦИИ СТРАТЕГИЙ
# =============================================================================

class CLIPStyleEmbedding(EmbeddingModel):
    """ЗАГЛУШКА: Замените на реальный вызов CLIP"""
    def encode(self, images: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        # Имитация батчинга для предотвращения OOM
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            # model.get_image_features(batch)
            embeddings.append(np.random.randn(len(batch), 512).astype(np.float32))
        return np.vstack(embeddings)

    @property
    def dim(self) -> int: return 512

class IsolationForestDetector(OutlierDetector):
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination

    def detect(self, embeddings: np.ndarray) -> np.ndarray:
        iso = IsolationForest(contamination=self.contamination, random_state=42)
        preds = iso.fit_predict(embeddings)
        return preds != -1  # True для нормальных, False для выбросов

class KMeansClusterer(Clusterer):
    def __init__(self, max_clusters: int = 20):
        self.max_clusters = max_clusters

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        n = len(embeddings)
        n_clusters = max(3, min(self.max_clusters, int(np.sqrt(n / 2))))
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(embeddings)


# =============================================================================
# 3. УТИЛИТЫ (ЗАГРУЗЧИК И ЭКСПОРТЕР)
# =============================================================================

class ImageLoader:
    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

    def __init__(self, directory: str):
        if cv2 is None: raise ImportError("Требуется OpenCV")
        # ИЗМЕНЕНИЕ: Сохраняем абсолютный путь к директории сразу
        self.directory = os.path.abspath(directory)

    def load(self) -> Tuple[List[np.ndarray], List[str]]:
        images, paths = [], []
        for path in self._scan_directory():
            img = cv2.imread(path)
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # ИЗМЕНЕНИЕ: Преобразуем путь к файлу в абсолютный
                paths.append(os.path.abspath(path))
        logger.info(f"Загружено {len(images)} изображений из {self.directory}")
        return images, paths

    def _scan_directory(self) -> List[str]:
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(glob.glob(os.path.join(self.directory, f"*{ext}")))
            files.extend(glob.glob(os.path.join(self.directory, f"*{ext.upper()}")))
        return files

class ResultExporter:
    @staticmethod
    def export(selected_paths: List[str], selected_embeddings: np.ndarray,
               selected_scores: np.ndarray, selected_clusters: np.ndarray, output_path: str):

        # Вычисление матрицы расстояний
        norm = selected_embeddings / (np.linalg.norm(selected_embeddings, axis=1, keepdims=True) + 1e-10)
        sim_matrix = norm @ norm.T
        np.fill_diagonal(sim_matrix, 1.0)
        dist_matrix = 1 - sim_matrix

        # Формирование обогащенного JSON
        selection_data = []
        for i, path in enumerate(selected_paths):
            selection_data.append({
                "file_path": path,  # Здесь теперь будет абсолютный путь
                "style_similarity": float(selected_scores[i]),
                "cluster_id": int(selected_clusters[i])
            })

        result = {
            "metadata": {
                "total_selected": len(selected_paths),
                "avg_similarity": float(np.mean(selected_scores))
            },
            "selection": selection_data,
            "distance_matrix": dist_matrix.tolist()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        logger.info(f"Результат сохранен в {output_path}")


# =============================================================================
# 4. ЯДРО ЛОГИКИ ОТБОРА (Чистые функции/статика)
# =============================================================================

class MMRSelector:
    """Maximal Marginal Relevance алгоритм отбора"""

    @staticmethod
    def select(embeddings: np.ndarray, cluster_labels: np.ndarray,
               similarity_scores: np.ndarray, criteria: SelectionCriteria,
               target_size: int) -> List[int]:

        n = len(embeddings)
        dist_matrix = MMRSelector._compute_distance_matrix(embeddings)

        # 1. Гарантируем покрытие кластеров
        selected = set()
        for lab in np.unique(cluster_labels):
            idx = np.where(cluster_labels == lab)[0]
            selected.add(int(idx[np.argmax(similarity_scores[idx])]))

        # 2. Жадный MMR добор
        remaining = set(range(n)) - selected
        selected_list = list(selected)

        while len(selected_list) < target_size and remaining:
            rem_arr = np.array(list(remaining))
            sel_arr = np.array(selected_list)

            min_dists = np.min(dist_matrix[np.ix_(rem_arr, sel_arr)], axis=1)
            mmr = (criteria.similarity_weight * similarity_scores[rem_arr] +
                   criteria.diversity_weight * min_dists)

            best = rem_arr[np.argmax(mmr)]
            selected_list.append(int(best))
            remaining.remove(best)

        return selected_list

    @staticmethod
    def _compute_distance_matrix(features: np.ndarray) -> np.ndarray:
        norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
        sim = norm @ norm.T
        np.fill_diagonal(sim, 1.0)
        return 1 - sim


# =============================================================================
# 5. ОРКЕСТРАТОР ПАЙПЛАЙНА (FACADE)
# =============================================================================

class LoRADatasetCurator:
    def __init__(self, model: EmbeddingModel, detector: OutlierDetector,
                 clusterer: Clusterer, criteria: SelectionCriteria, target_size: int):
        self.model = model
        self.detector = detector
        self.clusterer = clusterer
        self.criteria = criteria
        self.target_size = target_size

    def process(self, input_dir: str, reference_dir: Optional[str], output_json: str):
        # 1. Загрузка данных
        loader = ImageLoader(input_dir)
        images, image_paths = loader.load()
        if not images: return

        # 2. Загрузка референсов (если есть)
        ref_embeddings = None
        if reference_dir:
            ref_loader = ImageLoader(reference_dir)
            ref_images, _ = ref_loader.load()
            if ref_images:
                logger.info(f"Вычисление эмбеддингов для {len(ref_images)} референсов...")
                ref_embeddings = self.model.encode(ref_images)

        # 3. Эмбеддинги основного датасета
        logger.info("Вычисление эмбеддингов датасета батчами...")
        all_embeddings = self.model.encode(images, batch_size=32)

        if len(images) <= self.target_size:
            logger.info("Количество изображений <= target_size. Возврат всех файлов.")
            scores = self._compute_scores(all_embeddings, ref_embeddings)
            ResultExporter.export(image_paths, all_embeddings, scores, np.zeros(len(images)), output_json)
            return

        # 4. Фильтрация выбросов
        logger.info("Фильтрация выбросов...")
        valid_mask = self.detector.detect(all_embeddings)
        valid_embeddings = all_embeddings[valid_mask]
        valid_paths = [image_paths[i] for i in range(len(images)) if valid_mask[i]]

        if len(valid_paths) <= self.target_size:
            logger.info(f"После фильтрации осталось {len(valid_paths)} <= target_size.")
            scores = self._compute_scores(valid_embeddings, ref_embeddings)
            ResultExporter.export(valid_paths, valid_embeddings, scores, np.zeros(len(valid_paths)), output_json)
            return

        # 5. Кластеризация
        logger.info("Кластеризация...")
        cluster_labels = self.clusterer.cluster(valid_embeddings)

        # 6. Скоринг сходства
        logger.info("Вычисление скоров сходства со стилем...")
        sim_scores = self._compute_scores(valid_embeddings, ref_embeddings)

        # 7. MMR Отбор
        logger.info(f"Отбор {self.target_size} изображений...")
        selected_indices = MMRSelector.select(
            embeddings=valid_embeddings,
            cluster_labels=cluster_labels,
            similarity_scores=sim_scores,
            criteria=self.criteria,
            target_size=self.target_size
        )

        # 8. Экспорт
        ResultExporter.export(
            selected_paths=[valid_paths[i] for i in selected_indices],
            selected_embeddings=valid_embeddings[selected_indices],
            selected_scores=sim_scores[selected_indices],
            selected_clusters=cluster_labels[selected_indices],
            output_path=output_json
        )

    def _compute_scores(self, embeddings: np.ndarray, ref_embeddings: Optional[np.ndarray]) -> np.ndarray:
        """Вычисляет сходство с референсами или с центроидом датасета"""
        if ref_embeddings is not None:
            ref_center = ref_embeddings.mean(axis=0, keepdims=True)
            return cosine_similarity(embeddings, ref_center).flatten()
        else:
            logger.warning("Референсы не заданы. Скоринг ведется относительно центра датасета.")
            center = embeddings.mean(axis=0, keepdims=True)
            return cosine_similarity(embeddings, center).flatten()


# =============================================================================
# 6. КОНСОЛЬНЫЙ ИНТЕРФЕЙС
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ООП Система отбора датасета для LoRA.")
    parser.add_argument("--input-dir", type=str, required=True, help="Папка с исходными картинками")
    parser.add_argument("--target-size", type=int, required=True, help="Желаемое количество картинок")
    parser.add_argument("--output-json", type=str, default="result.json", help="Путь к JSON файлу")
    parser.add_argument("--reference-dir", type=str, default=None, help="Папка с эталонными картинками стиля (опционально)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Ошибка: Директория '{args.input_dir}' не найдена.")
        return

    # Сборка пайплайна (Composition Root)
    model = CLIPStyleEmbedding()
    detector = IsolationForestDetector(contamination=0.1)
    clusterer = KMeansClusterer(max_clusters=20)
    criteria = SelectionCriteria()

    curator = LoRADatasetCurator(
        model=model,
        detector=detector,
        clusterer=clusterer,
        criteria=criteria,
        target_size=args.target_size
    )

    curator.process(
        input_dir=args.input_dir,
        reference_dir=args.reference_dir,
        output_json=args.output_json
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()