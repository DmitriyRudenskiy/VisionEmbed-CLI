import argparse
import gc
import json
import logging
import math
import os
import signal
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "microsoft/VibeVoice-ASR-HF"
SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 30.0          # длительность одного чанка при forced chunking
OVERLAP_DURATION_SEC = 2.0         # overlap между чанками
MAX_SUBTITLE_CHARS = 42            # макс. длина строки субтитров
MIN_SEGMENT_DURATION = 0.5         # мин. длительность сегмента (сек)
FORCE_CHUNKING_THRESHOLD_SEC = 1800.0  # 30 мин — принудительное чанкование
CHECKPOINT_EXT = ".checkpoint.json"
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("vibevoice_asr")


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    fmt = "[%(asctime)s] %(levelname)-8s %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.setLevel(level)
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning("Получен сигнал прерывания. Завершение после текущего чанка...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def format_timestamp_srt(seconds: float) -> str:
    """Конвертирует секунды в формат SRT: ЧЧ:ММ:СС,мс"""
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Конвертирует секунды в формат VTT: ЧЧ:ММ:СС.мс"""
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def validate_chunk_size(value: int) -> int:
    if value % 3200 != 0:
        raise argparse.ArgumentTypeError(
            f"chunk_size должен быть кратен 3200 (получено {value}). "
            f"Ближайшие допустимые значения: {value - value % 3200}, {value + (3200 - value % 3200)}"
        )
    return value


def get_audio_duration(audio_path: str) -> float:
    """Возвращает длительность аудио в секундах."""
    try:
        duration = librosa.get_duration(path=audio_path)
        return float(duration)
    except Exception as e:
        logger.warning(f"Не удалось определить длительность через librosa ({e}), пробуем загрузить...")
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return float(len(y)) / sr


def split_long_text(text: str, max_chars: int = MAX_SUBTITLE_CHARS) -> List[str]:
    """Разбивает длинный текст на строки не более max_chars символов по словам."""
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            if current:
                lines.append(current.strip())
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current.strip())
    return lines


def merge_short_segments(
    segments: List[Dict[str, Any]], min_duration: float = MIN_SEGMENT_DURATION
) -> List[Dict[str, Any]]:
    """Объединяет слишком короткие сегменты (< min_duration) с соседними."""
    if not segments:
        return []
    merged: List[Dict[str, Any]] = [segments[0].copy()]
    for seg in segments[1:]:
        duration = seg.get("End", 0.0) - seg.get("Start", 0.0)
        last = merged[-1]
        last_duration = last.get("End", 0.0) - last.get("Start", 0.0)
        same_speaker = seg.get("Speaker") == last.get("Speaker")
        if duration < min_duration and same_speaker:
            last["End"] = seg.get("End", last["End"])
            last["Content"] = f"{last.get('Content', '')} {seg.get('Content', '')}".strip()
        else:
            merged.append(seg.copy())
    return merged


def filter_silence(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Удаляет сегменты, помеченные как тишина."""
    filtered = []
    for seg in segments:
        text = seg.get("Content", "").strip()
        if text and text.lower() not in {"[silence]", "[silence]", "(silence)", "<silence>"}:
            filtered.append(seg)
    return filtered


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_checkpoint(checkpoint_path: str) -> Optional[Tuple[int, List[Dict[str, Any]]]]:
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        next_chunk = data.get("next_chunk", 0)
        segments = data.get("segments", [])
        logger.info(f"Загружен checkpoint: {len(segments)} сегментов, следующий чанк {next_chunk}")
        return next_chunk, segments
    except Exception as e:
        logger.warning(f"Не удалось загрузить checkpoint ({e}), начинаем сначала.")
        return None


def save_checkpoint(checkpoint_path: str, next_chunk: int, segments: List[Dict[str, Any]]) -> None:
    data = {"next_chunk": next_chunk, "segments": segments}
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------
def to_srt(segments: List[Dict[str, Any]], output_path: str) -> None:
    lines: List[str] = []
    idx = 1
    for seg in segments:
        text = seg.get("Content", "").strip()
        if not text:
            continue
        start = format_timestamp_srt(seg.get("Start", 0.0))
        end = format_timestamp_srt(seg.get("End", 0.0))
        speaker = seg.get("Speaker")
        if speaker is not None:
            text = f"[Speaker {speaker}]: {text}"
        for sub_line in split_long_text(text):
            lines.append(str(idx))
            lines.append(f"{start} --> {end}")
            lines.append(sub_line)
            lines.append("")
            idx += 1
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def to_vtt(segments: List[Dict[str, Any]], output_path: str) -> None:
    lines = ["WEBVTT", ""]
    for seg in segments:
        text = seg.get("Content", "").strip()
        if not text:
            continue
        start = format_timestamp_vtt(seg.get("Start", 0.0))
        end = format_timestamp_vtt(seg.get("End", 0.0))
        speaker = seg.get("Speaker")
        if speaker is not None:
            text = f"<v Speaker {speaker}>{text}</v>"
        lines.append(f"{start} --> {end}")
        for sub_line in split_long_text(text):
            lines.append(sub_line)
        lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def to_json(segments: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def to_txt(segments: List[Dict[str, Any]], output_path: str) -> None:
    lines = []
    for seg in segments:
        text = seg.get("Content", "").strip()
        if not text:
            continue
        speaker = seg.get("Speaker")
        if speaker is not None:
            text = f"[Speaker {speaker}]: {text}"
        lines.append(text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main transcriber class
# ---------------------------------------------------------------------------
class VibeVoiceTranscriber:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        chunk_size: int = 1440000,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: str = "auto",
    ):
        self.model_id = model_id
        self.chunk_size = chunk_size
        self.device_map = device_map

        if torch_dtype is None:
            if torch.cuda.is_available():
                self.torch_dtype = torch.bfloat16
            elif torch.backends.mps.is_available():
                self.torch_dtype = torch.float32
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype

        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        self._load_model()

    def _load_model(self) -> None:
        logger.info(f"Загрузка модели {self.model_id}...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            logger.info(f"Модель загружена, dtype={self.model.dtype}, device={self.model.device}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise

    def _transcribe_chunk(
        self,
        audio_path: str,
        prompt: Optional[str] = None,
        offset_sec: float = 0.0,
        duration_sec: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Транскрибирует один чанк аудио."""
        try:
            if duration_sec is not None:
                y, sr = librosa.load(
                    audio_path,
                    sr=SAMPLE_RATE,
                    mono=True,
                    offset=offset_sec,
                    duration=duration_sec,
                )
                # librosa возвращает numpy array, сохраняем во временный wav
                import tempfile
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, y, sr)
                    tmp_path = tmp.name
                inputs = self.processor.apply_transcription_request(
                    audio=tmp_path,
                    prompt=prompt,
                ).to(self.model.device, dtype=self.model.dtype)
                os.unlink(tmp_path)
            else:
                inputs = self.processor.apply_transcription_request(
                    audio=audio_path,
                    prompt=prompt,
                ).to(self.model.device, dtype=self.model.dtype)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    tokenizer_chunk_size=self.chunk_size,
                )

            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, prompt_len:]
            transcription = self.processor.decode(generated_ids, return_format="parsed")[0]

            if isinstance(transcription, str):
                logger.warning(f"Модель вернула строку вместо списка: {transcription[:200]}")
                return []

            # Корректируем таймстампы с учётом offset
            for seg in transcription:
                seg["Start"] = seg.get("Start", 0.0) + offset_sec
                seg["End"] = seg.get("End", 0.0) + offset_sec

            return transcription

        except torch.cuda.OutOfMemoryError:
            logger.error("OOM при обработке чанка. Попробуйте уменьшить --chunk_size.")
            raise
        except Exception as e:
            logger.error(f"Ошибка транскрипции чанка (offset={offset_sec}): {e}")
            raise

    def transcribe(
        self,
        audio_path: str,
        prompt: Optional[str] = None,
        output_path: Optional[str] = None,
        output_format: str = "srt",
        resume: bool = False,
        force_chunking: bool = False,
    ) -> str:
        """
        Полный pipeline транскрипции с автоматическим чанкованием.
        Возвращает путь к сохранённому файлу.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Файл не найден: {audio_path}")

        # Определяем выходной путь
        if output_path:
            out_path = output_path
        else:
            out_path = os.path.splitext(audio_path)[0] + f".{output_format}"

        checkpoint_path = out_path + CHECKPOINT_EXT
        duration = get_audio_duration(audio_path)
        logger.info(f"Длительность аудио: {duration:.1f} сек ({duration/60:.1f} мин)")

        # Определяем, нужно ли чанкование
        need_chunking = force_chunking or duration > FORCE_CHUNKING_THRESHOLD_SEC

        if not need_chunking:
            # Пробуем целиком
            try:
                logger.info("Распознавание целиком (без чанкования)...")
                segments = self._transcribe_chunk(audio_path, prompt=prompt)
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM на полном файле, переключаемся на чанкование...")
                need_chunking = True
                torch.cuda.empty_cache()
                gc.collect()

        all_segments: List[Dict[str, Any]] = []
        start_chunk = 0

        if need_chunking:
            # Проверяем checkpoint
            if resume:
                ckpt = load_checkpoint(checkpoint_path)
                if ckpt:
                    start_chunk, all_segments = ckpt

            chunk_samples = int(CHUNK_DURATION_SEC * SAMPLE_RATE)
            overlap_samples = int(OVERLAP_DURATION_SEC * SAMPLE_RATE)
            total_samples = int(duration * SAMPLE_RATE)
            num_chunks = math.ceil((total_samples - overlap_samples) / (chunk_samples - overlap_samples))

            logger.info(f"Чанкование: {num_chunks} чанков по ~{CHUNK_DURATION_SEC}с с overlap {OVERLAP_DURATION_SEC}с")

            pbar = tqdm(total=num_chunks, desc="Чанки", initial=start_chunk)
            for i in range(start_chunk, num_chunks):
                if _shutdown_requested:
                    logger.warning("Прерывание по запросу пользователя.")
                    save_checkpoint(checkpoint_path, i, all_segments)
                    logger.info(f"Checkpoint сохранён: {checkpoint_path}")
                    break

                offset = i * (CHUNK_DURATION_SEC - OVERLAP_DURATION_SEC)
                dur = CHUNK_DURATION_SEC + OVERLAP_DURATION_SEC if i < num_chunks - 1 else None

                try:
                    segs = self._transcribe_chunk(audio_path, prompt=prompt, offset_sec=offset, duration_sec=dur)
                except torch.cuda.OutOfMemoryError:
                    logger.error("OOM даже на одном чанке. Попробуйте ещё сильнее уменьшить --chunk_size.")
                    save_checkpoint(checkpoint_path, i, all_segments)
                    raise

                # Удаляем дубли в overlap-зоне (простая эвристика: если start < offset + overlap/2 и уже есть похожий текст)
                if i > 0 and overlap_duration > 0:
                    cutoff = offset + OVERLAP_DURATION_SEC / 2
                    segs = [s for s in segs if s.get("Start", 0.0) >= cutoff]

                all_segments.extend(segs)
                save_checkpoint(checkpoint_path, i + 1, all_segments)
                pbar.update(1)

                # Очистка памяти
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            pbar.close()

        # Пост-обработка
        logger.info(f"Пост-обработка: {len(all_segments)} сырых сегментов")
        all_segments = filter_silence(all_segments)
        all_segments = merge_short_segments(all_segments)

        if not all_segments:
            logger.warning("Распознанный текст пуст.")
            return out_path

        # Сохранение
        formatters = {
            "srt": to_srt,
            "vtt": to_vtt,
            "json": to_json,
            "txt": to_txt,
        }
        if output_format not in formatters:
            raise ValueError(f"Неизвестный формат: {output_format}. Доступны: {list(formatters.keys())}")

        formatters[output_format](all_segments, out_path)
        logger.info(f"Сохранено: {out_path} ({len(all_segments)} сегментов)")

        # Удаляем checkpoint при успехе
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="VibeVoice ASR Transcriber — улучшенная версия с чанкованием, checkpoint и множеством форматов."
    )
    parser.add_argument("input", nargs="?", help="Путь к аудиофайлу")
    parser.add_argument("--input-dir", help="Путь к директории с аудиофайлами (batch-режим)")
    parser.add_argument("--prompt", default=None, help="Контекст / горячие слова")
    parser.add_argument(
        "--chunk-size",
        type=validate_chunk_size,
        default=1440000,
        help="Размер чанка для генерации (кратен 3200, по умолчанию 1440000 ~60сек)",
    )
    parser.add_argument("-o", "--output", default=None, help="Путь для сохранения результата")
    parser.add_argument(
        "--format",
        choices=["srt", "vtt", "json", "txt"],
        default="srt",
        help="Формат выходного файла (по умолчанию srt)",
    )
    parser.add_argument(
        "--force-chunking",
        action="store_true",
        help="Принудительно использовать чанкование даже для коротких файлов",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Возобновить прерванную транскрипцию из checkpoint",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Уровень логирования",
    )
    args = parser.parse_args()

    setup_logging(level=getattr(logging, args.log_level))

    if not args.input and not args.input_dir:
        parser.error("Укажите либо input (файл), либо --input-dir (директория)")

    transcriber = VibeVoiceTranscriber(chunk_size=args.chunk_size)

    files_to_process: List[str] = []
    if args.input:
        files_to_process.append(args.input)
    if args.input_dir:
        for ext in SUPPORTED_AUDIO_EXTS:
            files_to_process.extend(Path(args.input_dir).glob(f"*{ext}"))
            files_to_process.extend(Path(args.input_dir).glob(f"*{ext.upper()}"))
        files_to_process = [str(p) for p in files_to_process]
        files_to_process = sorted(set(files_to_process))
        logger.info(f"Найдено файлов для обработки: {len(files_to_process)}")

    for audio_path in files_to_process:
        try:
            transcriber.transcribe(
                audio_path=str(audio_path),
                prompt=args.prompt,
                output_path=args.output if len(files_to_process) == 1 else None,
                output_format=args.format,
                resume=args.resume,
                force_chunking=args.force_chunking,
            )
        except Exception as e:
            logger.error(f"Не удалось обработать {audio_path}: {e}")
            if len(files_to_process) == 1:
                sys.exit(1)


if __name__ == "__main__":
    main()