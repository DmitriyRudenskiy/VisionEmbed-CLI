#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DWPose Dataset Builder (TorchScript версия)
Скрипт для извлечения поз из кадров видео и построения курируемой базы данных.
Использует controlnet_aux с TorchScript моделями — без ONNX и MMPose.

Установка:
    pip install controlnet-aux opencv-python numpy torch torchvision pillow
    # Модели скачаются автоматически с HuggingFace при первом запуске
    # Или укажите путь к локальным .torchscript.pt файлам

Запуск:
    python pose_dataset_builder.py build -i /path/to/frames -o dataset.db
"""

import os
import sys
import json
import sqlite3
import argparse
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import cv2
from PIL import Image
import torch

# controlnet_aux — DWPose с TorchScript, без ONNX/MMPose
try:
    from controlnet_aux import DWposeDetector
except ImportError:
    print("Установите: pip install controlnet-aux")
    raise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PoseDB")


# =============================================================================
# Конфигурация правил подбора датасета
# =============================================================================

@dataclass
class DatasetRules:
    """Правила отбора кадров для обучения нейросети pose estimation."""
    min_keypoint_confidence: float = 0.3
    min_visible_body_ratio: float = 0.6
    min_pose_change_threshold: float = 0.08
    max_redundant_frames: int = 3
    normalization: str = 'bbox'
    inference_resolution: int = 1024
    include_hand: bool = True
    include_face: bool = False
    include_body: bool = True


# =============================================================================
# Структуры данных
# =============================================================================

@dataclass
class PoseFrame:
    frame_id: int
    file_path: str
    file_hash: str
    width: int
    height: int
    body_keypoints: np.ndarray
    left_hand_keypoints: Optional[np.ndarray]
    right_hand_keypoints: Optional[np.ndarray]
    face_keypoints: Optional[np.ndarray]
    bbox: Tuple[float, float, float, float]
    avg_confidence: float
    visible_ratio: float
    timestamp: Optional[float] = None
    normalized_pose_vector: Optional[np.ndarray] = None


# =============================================================================
# База данных SQLite
# =============================================================================

class PoseDatabase:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS frames (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        frame_id INTEGER NOT NULL,
        file_path TEXT UNIQUE NOT NULL,
        file_hash TEXT NOT NULL,
        width INTEGER, height INTEGER,
        bbox TEXT,
        body_kpts TEXT,
        left_hand_kpts TEXT,
        right_hand_kpts TEXT,
        face_kpts TEXT,
        avg_confidence REAL,
        visible_ratio REAL,
        normalized_pose_vector BLOB,
        pose_change_score REAL,
        is_selected INTEGER DEFAULT 0,
        selection_reason TEXT,
        timestamp REAL
    );
    CREATE TABLE IF NOT EXISTS dataset_stats (key TEXT PRIMARY KEY, value TEXT);
    CREATE INDEX IF NOT EXISTS idx_selected ON frames(is_selected);
    CREATE INDEX IF NOT EXISTS idx_frame_id ON frames(frame_id);
    """

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def insert_frame(self, frame: PoseFrame, is_selected: bool = False, reason: str = "",
                     pose_change: float = 0.0):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO frames (
                frame_id, file_path, file_hash, width, height, bbox,
                body_kpts, left_hand_kpts, right_hand_kpts, face_kpts,
                avg_confidence, visible_ratio, normalized_pose_vector,
                pose_change_score, is_selected, selection_reason, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            frame.frame_id, frame.file_path, frame.file_hash,
            frame.width, frame.height, json.dumps(frame.bbox),
            json.dumps(frame.body_keypoints.tolist()),
            json.dumps(frame.left_hand_keypoints.tolist()) if frame.left_hand_keypoints is not None else None,
            json.dumps(frame.right_hand_keypoints.tolist()) if frame.right_hand_keypoints is not None else None,
            json.dumps(frame.face_keypoints.tolist()) if frame.face_keypoints is not None else None,
            frame.avg_confidence, frame.visible_ratio,
            frame.normalized_pose_vector.tobytes() if frame.normalized_pose_vector is not None else None,
            pose_change, int(is_selected), reason, frame.timestamp
        ))
        self.conn.commit()
        return cur.lastrowid

    def get_last_selected_pose(self) -> Optional[PoseFrame]:
        row = self.conn.execute(
            "SELECT * FROM frames WHERE is_selected=1 ORDER BY frame_id DESC LIMIT 1"
        ).fetchone()
        return self._row_to_frame(row) if row else None

    def _row_to_frame(self, row: sqlite3.Row) -> PoseFrame:
        def load_kpts(field):
            return np.array(json.loads(field), dtype=np.float32) if field else None

        vec = row["normalized_pose_vector"]
        return PoseFrame(
            frame_id=row["frame_id"], file_path=row["file_path"],
            file_hash=row["file_hash"], width=row["width"], height=row["height"],
            body_keypoints=load_kpts(row["body_kpts"]),
            left_hand_keypoints=load_kpts(row["left_hand_kpts"]),
            right_hand_keypoints=load_kpts(row["right_hand_kpts"]),
            face_keypoints=load_kpts(row["face_kpts"]),
            bbox=tuple(json.loads(row["bbox"])),
            avg_confidence=row["avg_confidence"], visible_ratio=row["visible_ratio"],
            timestamp=row["timestamp"],
            normalized_pose_vector=np.frombuffer(vec, dtype=np.float32) if vec else None
        )

    def get_selected_frames(self) -> List[PoseFrame]:
        rows = self.conn.execute(
            "SELECT * FROM frames WHERE is_selected=1 ORDER BY frame_id"
        ).fetchall()
        return [self._row_to_frame(r) for r in rows]

    def set_stat(self, key: str, value: Union[str, int, float]):
        self.conn.execute(
            "INSERT OR REPLACE INTO dataset_stats (key, value) VALUES (?, ?)",
            (key, str(value))
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


# =============================================================================
# Извлечение позы через controlnet_aux DWPose (TorchScript)
# =============================================================================

class PoseExtractor:
    def __init__(self, rules: DatasetRules, device: str = "mps"):
        self.rules = rules
        self.device = device

        # Определяем лучший доступный device на macOS
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        logger.info(f"Инициализация DWPose (TorchScript) на {device} ...")

        # DWposeDetector из controlnet_aux автоматически скачает модели с HuggingFace
        # или использует локальные если указать пути
        self.detector = DWposeDetector.from_pretrained("lllyasviel/Annotators")
        self.detector.to(device)

        logger.info("DWPose готов.")

    def extract(self, image_path: Path, frame_id: int) -> Optional[PoseFrame]:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # DWposeDetector возвращает PIL Image со скелетом, но нам нужны сырые ключевые точки
        # Используем внутренний метод для получения словаря с ключевыми точками
        pose_dict = self._detect_pose(img)

        if pose_dict is None or not pose_dict.get("bodies"):
            logger.debug(f"Нет детекции на {image_path.name}")
            return None

        # Парсим результат
        bodies = pose_dict["bodies"]
        if isinstance(bodies, dict):
            candidate = np.array(bodies.get("candidate", []), dtype=np.float32)
            subset = np.array(bodies.get("subset", []), dtype=np.float32)
        else:
            candidate = np.array(bodies, dtype=np.float32)
            subset = None

        if candidate.size == 0:
            return None

        # Выбираем первого (лучшего) человека
        if candidate.ndim == 2:
            candidate = candidate[np.newaxis, ...]
            if subset is not None and subset.ndim == 1:
                subset = subset[np.newaxis, ...]

        person_idx = 0
        if candidate.shape[0] > 1 and subset is not None:
            visible = (subset[:, :18] > 0).sum(axis=1)
            person_idx = int(np.argmax(visible))

        person_kpts = candidate[person_idx]  # (N_kpts, 3) или (N_kpts, 2)

        # Добавляем канал confidence если его нет
        if person_kpts.shape[1] == 2:
            confs = np.ones((person_kpts.shape[0], 1), dtype=np.float32) * 0.8
            person_kpts = np.concatenate([person_kpts, confs], axis=1)

        # Разделение на body/hands/face по формату COCO-WholeBody
        n_body = min(23, person_kpts.shape[0])
        body_kpts = person_kpts[:n_body].copy()

        # Применяем subset-маску если есть
        if subset is not None:
            for i in range(n_body):
                if subset[person_idx, i] < 0:
                    body_kpts[i, 2] = 0.0

        left_hand = right_hand = face = None

        # Руки и лицо обычно идут после body в COCO-WholeBody
        if self.rules.include_hand and person_kpts.shape[0] > n_body + 21:
            lh_start = n_body
            rh_start = n_body + 21
            left_hand = person_kpts[lh_start:lh_start + 21].copy()
            right_hand = person_kpts[rh_start:rh_start + 21].copy()

        if self.rules.include_face and person_kpts.shape[0] > n_body + 42 + 68:
            face_start = n_body + 42
            face = person_kpts[face_start:face_start + 68].copy()

        # Bounding box по видимым точкам
        visible = body_kpts[body_kpts[:, 2] > self.rules.min_keypoint_confidence]
        if len(visible) == 0:
            return None

        xs, ys = visible[:, 0], visible[:, 1]
        bbox = (float(xs.min()), float(ys.min()),
                float(xs.max() - xs.min()), float(ys.max() - ys.min()))

        confs = body_kpts[:, 2]
        avg_conf = float(np.mean(confs[confs > 0])) if np.any(confs > 0) else 0.0
        visible_ratio = float(np.mean(confs > self.rules.min_keypoint_confidence))

        file_hash = hashlib.md5(open(image_path, "rb").read(8192)).hexdigest()

        return PoseFrame(
            frame_id=frame_id, file_path=str(image_path), file_hash=file_hash,
            width=w, height=h, body_keypoints=body_kpts,
            left_hand_keypoints=left_hand, right_hand_keypoints=right_hand,
            face_keypoints=face, bbox=bbox,
            avg_confidence=avg_conf, visible_ratio=visible_ratio
        )

    def _detect_pose(self, img: Image.Image) -> Optional[Dict]:
        """Внутренний метод для получения сырых ключевых точек."""
        # Конвертируем PIL в numpy для detector
        img_np = np.array(img)

        # Используем внутренний метод DWposeDetector для получения словаря
        # Примечание: API может варьироваться, адаптируем под реальность
        try:
            # Пробуем получить pose dict через внутренний preprocess
            if hasattr(self.detector, 'detect_pose'):
                return self.detector.detect_pose(img_np)
            elif hasattr(self.detector, '__call__'):
                # Если возвращает только изображение, используем альтернативный путь
                result = self.detector(img, output_type="dict")
                return result if isinstance(result, dict) else None
        except Exception as e:
            logger.warning(f"Ошибка детекции: {e}")
            return None

        return None


# =============================================================================
# Курирование датасета
# =============================================================================

class DatasetCurator:
    def __init__(self, rules: DatasetRules):
        self.rules = rules
        self.last_selected_vector: Optional[np.ndarray] = None
        self.redundant_count: int = 0

    def normalize_pose(self, frame: PoseFrame) -> np.ndarray:
        kpts = frame.body_keypoints.copy()
        valid = kpts[:, 2] > self.rules.min_keypoint_confidence

        if self.rules.normalization == 'bbox':
            x, y, bw, bh = frame.bbox
            if bw < 1e-6 or bh < 1e-6:
                bw, bh = 1.0, 1.0
            kpts[:, 0] = (kpts[:, 0] - x) / bw
            kpts[:, 1] = (kpts[:, 1] - y) / bh
        elif self.rules.normalization == 'torso':
            torso_idx = [5, 6, 11, 12]
            torso_pts = kpts[torso_idx]
            torso_valid = torso_pts[:, 2] > 0
            if np.any(torso_valid):
                valid_torso = torso_pts[torso_valid][:, :2]
                center = valid_torso.mean(axis=0)
                scale = np.linalg.norm(valid_torso.max(axis=0) - valid_torso.min(axis=0))
                scale = max(scale, 1e-6)
                kpts[:, 0] = (kpts[:, 0] - center[0]) / scale
                kpts[:, 1] = (kpts[:, 1] - center[1]) / scale
            else:
                kpts[:, 0] /= frame.width
                kpts[:, 1] /= frame.height
        else:
            kpts[:, 0] /= frame.width
            kpts[:, 1] /= frame.height

        kpts[~valid, :2] = 0.0
        return kpts[:, :2].flatten().astype(np.float32)

    def compute_pose_change(self, frame: PoseFrame) -> float:
        frame.normalized_pose_vector = self.normalize_pose(frame)
        if self.last_selected_vector is None:
            return float('inf')
        diff = np.linalg.norm(frame.normalized_pose_vector - self.last_selected_vector)
        return float(diff / max(len(frame.normalized_pose_vector) // 2, 1))

    def decide_selection(self, frame: PoseFrame, pose_change: float) -> Tuple[bool, str]:
        if frame.visible_ratio < self.rules.min_visible_body_ratio:
            return False, f"low_visibility:{frame.visible_ratio:.2f}"
        if frame.avg_confidence < self.rules.min_keypoint_confidence:
            return False, f"low_confidence:{frame.avg_confidence:.2f}"
        if pose_change < self.rules.min_pose_change_threshold:
            self.redundant_count += 1
            if self.redundant_count >= self.rules.max_redundant_frames:
                return False, f"redundant:change={pose_change:.4f}"
            return False, f"too_similar:change={pose_change:.4f}"
        else:
            self.redundant_count = 0
        if pose_change > 0.5:
            return False, f"jump_detected:change={pose_change:.4f}"
        return True, f"selected:change={pose_change:.4f}"

    def update_last(self, frame: PoseFrame):
        self.last_selected_vector = frame.normalized_pose_vector.copy()


# =============================================================================
# Основной пайплайн
# =============================================================================

def build_pose_database(input_dir: str, output_db: str,
                        rules: Optional[DatasetRules] = None,
                        device: str = "mps"):
    rules = rules or DatasetRules()
    input_path = Path(input_dir)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    frames = sorted([
        p for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    ], key=lambda p: p.name)

    if not frames:
        logger.error(f"Не найдено изображений в {input_dir}")
        return

    logger.info(f"Найдено {len(frames)} кадров")

    db = PoseDatabase(output_db)
    extractor = PoseExtractor(rules, device=device)
    curator = DatasetCurator(rules)

    total = len(frames)
    selected_count = 0

    for i, img_path in enumerate(frames, 1):
        logger.info(f"[{i}/{total}] {img_path.name}")

        frame = extractor.extract(img_path, frame_id=i)
        if frame is None:
            db.insert_frame(
                PoseFrame(i, str(img_path), "", 0, 0, np.zeros((1, 3)), None, None, None,
                          (0, 0, 0, 0), 0, 0),
                is_selected=False, reason="no_detection", pose_change=0.0
            )
            continue

        pose_change = curator.compute_pose_change(frame)
        is_selected, reason = curator