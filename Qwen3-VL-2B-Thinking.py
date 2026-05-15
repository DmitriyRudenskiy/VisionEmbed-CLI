import re
import json
import logging
import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Set, List, Tuple
from dataclasses import dataclass, asdict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ─── Логирование ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Конфигурация модели ───
@dataclass
class ModelConfig:
    """Все параметры загрузки и генерации модели."""
    model_name: str = "Qwen/Qwen3-VL-2B-Thinking"
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True
    padding_side: str = "left"

    max_new_tokens: int = 4096
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.0

    def get_torch_dtype(self):
        mapping = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(self.torch_dtype, "auto")

    def generation_kwargs(self) -> dict:
        return dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )


# ─── Результат обработки ───
@dataclass
class ProcessingResult:
    image: str
    prediction: str                # финальный ответ после </think>
    thinking: str = ""             # цепочка рассуждений до </think>
    analysis_time_sec: float = 0.0
    batch_tokens_per_sec: float = 0.0


# ─── Очистка текста ───
class TextCleaner:
    """Нормализует текст, сохраняя теги <think> и </think>."""
    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase
        self.allowed_pattern = re.compile(r"[^\w\s.,:;!?()\-\[\]\'\"«»„“/]", re.UNICODE)
        self.split_re = re.compile(r"</think\s*>", re.IGNORECASE)

    def clean(self, text: str) -> str:
        """Удаляет недопустимые символы, нормализует пробелы. Теги не трогает."""
        text = self.allowed_pattern.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        if self.lowercase:
            text = text.lower()
        return text

    def split_thinking_and_answer(self, raw_text: str) -> Tuple[str, str]:
        """
        Разбивает вывод по закрывающему тегу </think>.
        Возвращает (thinking, answer), где thinking – текст до </think> с удалёнными тегами,
        answer – текст после </think>. Если тега нет, thinking = "".
        """
        parts = self.split_re.split(raw_text, maxsplit=1)
        if len(parts) == 2:
            thinking = parts[0].strip()
            # убираем открывающий <think> с возможными атрибутами
            thinking = re.sub(r"<think\b[^>]*>", "", thinking, flags=re.IGNORECASE).strip()
            answer = parts[1].strip()
        else:
            thinking = ""
            answer = parts[0].strip()

        # Применяем стандартную чистку (тегов уже нет)
        thinking = self.clean(thinking)
        answer = self.clean(answer)
        return thinking, answer


# ─── Сканер изображений ───
def scan_images(directory: Path, extensions: Set[str]) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Директория не найдена: {directory.resolve()}")
    images = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    return sorted(images)


# ─── Менеджер модели ───
class ModelInference:
    def __init__(self, config: ModelConfig):
        self.config = config
        logger.info(f"Загрузка модели {config.model_name}...")
        torch_dtype = config.get_torch_dtype()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=torch_dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
        )
        self.processor = AutoProcessor.from_pretrained(
            config.model_name, trust_remote_code=config.trust_remote_code
        )
        self.processor.tokenizer.padding_side = config.padding_side
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.model.eval()
        self._warmup()
        self._log_memory_usage()
        logger.info("Модель готова к работе.")

    def _warmup(self):
        logger.info("Прогрев модели...")
        try:
            dummy_img = Image.new('RGB', (64, 64), color='black')
            messages = [[{"role": "user", "content": [
                {"type": "image", "image": dummy_img},
                {"type": "text", "text": "init"}
            ]}]]
            text = self.processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[dummy_img], return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=1)
            logger.info("Прогрев завершён.")
        except Exception as e:
            logger.warning(f"Прогрев не удался: {e}")

    def _log_memory_usage(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"GPU {i}: allocated {allocated:.2f} GiB, reserved {reserved:.2f} GiB")

    @torch.no_grad()
    def predict(self, image_paths: List[str], prompt_text: str, cleaner: TextCleaner) -> List[ProcessingResult]:
        if not image_paths:
            return []
        images = []
        valid_paths = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                logger.error(f"Пропуск {path}: {e}")
        if not images:
            return []

        messages = [[{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt_text}
        ]}] for img in images]
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.model.device)

        start_time = time.time()
        generated_ids = self.model.generate(**inputs, **self.config.generation_kwargs())
        duration = time.time() - start_time

        input_lens = [len(in_ids) for in_ids in inputs.input_ids]
        trimmed_ids = [out_ids[in_len:] for out_ids, in_len in zip(generated_ids, input_lens)]
        raw_texts = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        total_tokens = sum(len(ids) for ids in trimmed_ids)
        batch_tps = total_tokens / duration if duration > 0 else 0.0

        results = []
        for path, raw in zip(valid_paths, raw_texts):
            thinking, answer = cleaner.split_thinking_and_answer(raw)
            results.append(ProcessingResult(
                image=path,
                prediction=answer,
                thinking=thinking,
                analysis_time_sec=round(duration, 4),
                batch_tokens_per_sec=round(batch_tps, 2),
            ))
        return results


# ─── Вспомогательные функции сохранения ───
def atomic_write_json(data: list, path: Path):
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

def append_text_lines(lines: List[str], path: Path):
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# ─── Основная логика ───
def main():
    parser = argparse.ArgumentParser(description="Batch image captioning with Qwen3-VL")
    parser.add_argument("directory", type=str, help="Папка с изображениями")
    parser.add_argument("--output", type=str, default=None, help="Путь к JSON‑результатам")
    parser.add_argument("--prompt", default="Create a descriptive detailed caption for this image.", help="Промпт")
    parser.add_argument("--extensions", default=".jpg,.jpeg,.png,.bmp,.gif,.webp,.tiff,.tif")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lowercase", action="store_true", help="Приводить описания к нижнему регистру")
    parser.add_argument("--no-resume", action="store_true", help="Начать заново, игнорируя существующий JSON")

    parser.add_argument("--model", dest="model_name", default="Qwen/Qwen3-VL-2B-Thinking")
    parser.add_argument("--device-map", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--greedy", action="store_true", help="Жадное декодирование")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()

    config = ModelConfig(
        model_name=args.model_name,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_tokens,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    target_dir = Path(args.directory).resolve()
    out_path = Path(args.output).resolve() if args.output else target_dir / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    desc_path = out_path.with_name("descriptions.txt")

    logger.info(f"Директория: {target_dir}")
    logger.info(f"JSON-вывод: {out_path}")
    logger.info(f"Текстовый вывод: {desc_path}")

    interrupted = False
    def shutdown_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        logger.warning("Получен сигнал остановки, сохраняем прогресс...")

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    extensions = set(ext.strip() for ext in args.extensions.split(","))
    try:
        all_image_paths = scan_images(target_dir, extensions)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    total_found = len(all_image_paths)
    logger.info(f"Найдено изображений: {total_found}")
    if total_found == 0:
        logger.warning("Изображения не найдены, завершение.")
        sys.exit(0)

    processed = set()
    existing_results = []
    if not args.no_resume and out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            if isinstance(existing_results, list):
                processed = {item.get("image") for item in existing_results if "image" in item}
                logger.info(f"Возобновление: {len(processed)} уже обработано")
        except Exception as e:
            logger.error(f"Ошибка чтения {out_path}: {e}, начинаем с чистого листа")
            existing_results = []

    if existing_results:
        logger.info("Пересоздание текстового файла с описаниями из JSON...")
        try:
            with open(desc_path, "w", encoding="utf-8") as f:
                for item in existing_results:
                    f.write(item.get("prediction", "") + "\n")
        except Exception as e:
            logger.error(f"Не удалось записать описания: {e}")

    to_process = [str(p) for p in all_image_paths if str(p) not in processed]
    if not to_process:
        logger.info("Все изображения уже обработаны.")
        sys.exit(0)

    cleaner = TextCleaner(lowercase=args.lowercase)
    model = ModelInference(config=config)
    results = existing_results.copy()
    batch_size = max(1, args.batch_size)

    for start_idx in tqdm(range(0, len(to_process), batch_size), desc="Обработка"):
        if interrupted:
            break
        batch_paths = to_process[start_idx:start_idx + batch_size]
        batch_results = model.predict(batch_paths, args.prompt, cleaner)
        if not batch_results:
            continue

        results.extend(asdict(r) for r in batch_results)
        try:
            atomic_write_json(results, out_path)
        except Exception as e:
            logger.error(f"Ошибка сохранения JSON: {e}")
        try:
            predictions = [r.prediction for r in batch_results]
            append_text_lines(predictions, desc_path)
        except Exception as e:
            logger.error(f"Ошибка записи текстового файла: {e}")

    logger.info(f"Обработано {len(to_process)} файлов (всего {len(results)} записей).")
    logger.info(f"Результаты: {out_path}")
    logger.info(f"Текстовые описания: {desc_path}")


if __name__ == "__main__":
    main()