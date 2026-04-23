import os
import re
import json
import logging
import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Set, List, Dict
from dataclasses import dataclass, asdict, field
from PIL import Image
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ─── Логирование ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Данные ───
@dataclass
class ProcessingResult:
    image: str
    prediction: str
    analysis_time_sec: float = 0.0
    tokens_per_sec: float = 0.0


# ─── Единая конфигурация модели ───
@dataclass
class ModelConfig:
    """
    Класс, содержащий все настройки для загрузки и работы модели.
    """
    # Идентификация
    model_name: str = "Qwen/Qwen3-VL-2B-Thinking"

    # Параметры загрузки
    device_map: str = "auto"
    torch_dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    trust_remote_code: bool = True
    padding_side: str = "left"  # Важно для батчинга LLM

    # Параметры генерации
    max_new_tokens: int = 4096
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.0


# ─── Очистка текста ───
class TextCleaner:
    def __init__(self, lowercase: bool = False, remove_thinking: bool = True):
        self.lowercase = lowercase
        self.remove_thinking = remove_thinking
        self.allowed_pattern = re.compile(r"[^\w\s.,:;!?()\-\[\]\'\"«»„“/]", re.UNICODE)
        self.thinking_pattern = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)

    def clean(self, raw_text: str) -> str:
        text = raw_text
        if self.remove_thinking:
            text = self.thinking_pattern.sub("", text)

        cleaned = self.allowed_pattern.sub("", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if self.lowercase:
            cleaned = cleaned.lower()
        return cleaned


# ─── Сканер изображений ───
def scan_images(directory: Path, extensions: Set[str]) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Директория не найдена: {directory.resolve()}")

    images = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    return sorted(images)


# ─── Модель ───
class ModelInference:
    def __init__(self, config: ModelConfig):
        self.config = config
        logger.info(f"Загрузка модели {config.model_name}...")

        # Маппинг строкового типа в torch dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        torch_dtype = dtype_map.get(config.torch_dtype, "auto")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=torch_dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
        )
        self.processor = AutoProcessor.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code
        )

        # Настройка токенизатора согласно конфигу
        self.processor.tokenizer.padding_side = config.padding_side
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        self.model.eval()

        # Прогрев модели
        self._warmup()

        logger.info(f"Модель загружена (device_map={config.device_map}, padding={config.padding_side})")

    def _warmup(self):
        """Прогрев модели для исключения влияния холодного старта на тайминги."""
        logger.info("Прогрев модели (CUDA initialization)...")
        try:
            dummy_img = Image.new('RGB', (64, 64), color='black')
            messages = [[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": dummy_img},
                        {"type": "text", "text": "Init"},
                    ],
                }
            ]]
            text = self.processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[dummy_img], return_tensors="pt", padding=True).to(
                self.model.device)

            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=1)
            logger.info("Прогрев завершен.")
        except Exception as e:
            logger.warning(f"Не удалось выполнить прогрев: {e}")

    def predict(self, image_paths: List[str], prompt_text: str, cleaner: TextCleaner) -> List[ProcessingResult]:
        if not image_paths:
            return []

        results: List[ProcessingResult] = []
        images: List[Image.Image] = []
        valid_paths: List[str] = []

        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                logger.error(f"Ошибка загрузки изображения {path}: {e}")

        if not images:
            return results

        try:
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                for img in images
            ]

            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in messages
            ]

            inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            start_time = time.time()

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                )

            end_time = time.time()
            duration_sec = end_time - start_time

            trimmed_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            raw_texts = self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for i, (path, raw) in enumerate(zip(valid_paths, raw_texts)):
                cleaned = cleaner.clean(raw)
                num_tokens = len(trimmed_ids[i])
                tps = num_tokens / duration_sec if duration_sec > 0 else 0.0

                results.append(ProcessingResult(
                    image=path,
                    prediction=cleaned,
                    analysis_time_sec=round(duration_sec, 4),
                    tokens_per_sec=round(tps, 2)
                ))

        except Exception as e:
            logger.error(f"Ошибка инференса батча: {e}")

        return results


# ─── Main ───
def main():
    parser = argparse.ArgumentParser(description="Batch image captioning with Qwen3-VL")

    # Основные аргументы
    parser.add_argument("directory", type=str, help="Папка с изображениями")
    parser.add_argument("--output", type=str, default=None, help="Путь к файлу results.json")
    parser.add_argument(
        "--prompt",
        default="Create a descriptive detailed caption for this image.",
        help="Текст промпта для генерации"
    )
    parser.add_argument(
        "--extensions", default=".jpg,.jpeg,.png,.bmp,.gif,.webp,.tiff,.tif"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lowercase", action="store_true")

    # Аргументы конфигурации модели
    parser.add_argument("--model", dest="model_name", default="Qwen/Qwen3-VL-2B-Thinking")
    parser.add_argument("--device-map", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    # Аргументы генерации
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--greedy", action="store_true", help="Использовать жадное декодирование")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()

    # ─── Инициализация конфигурации ───
    config = ModelConfig(
        model_name=args.model_name,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_tokens,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )

    # Пути
    target_dir = Path(args.directory).resolve()
    out_path = Path(args.output).resolve() if args.output else target_dir / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Путь к текстовому файлу описаний
    txt_log_path = out_path.parent / "prompt.txt"

    logger.info(f"Сканирование директории: {target_dir}")

    # Graceful shutdown
    interrupted = False

    def _handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        logger.warning("Прерывание... Сохранение прогресса...")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    extensions = set(args.extensions.split(","))
    try:
        all_image_paths = scan_images(target_dir, extensions)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    total_found = len(all_image_paths)
    logger.info(f"Найдено изображений: {total_found}")

    if total_found == 0:
        logger.warning("Изображения не найдены.")
        sys.exit(0)

    # Resume logic
    processed_images_paths: Set[str] = set()
    existing_results: List[Dict] = []

    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
                if isinstance(existing_results, list):
                    processed_images_paths = {item.get("image") for item in existing_results}
                    logger.info(f"Пропуск уже обработанных: {len(processed_images_paths)}")
        except Exception as e:
            logger.error(f"Ошибка чтения файла результатов: {e}")

    image_paths_to_process = [str(p) for p in all_image_paths if str(p) not in processed_images_paths]

    if not image_paths_to_process:
        logger.info("Все изображения обработаны.")
        sys.exit(0)

    cleaner = TextCleaner(lowercase=args.lowercase)
    model = ModelInference(config=config)

    results = existing_results
    batch_size = max(1, args.batch_size)
    total_to_process = len(image_paths_to_process)

    # Определяем режим открытия текстового файла
    # Если начинаем с нуля (нет существующих результатов), перезаписываем файл ('w')
    # Если продолжаем работу, дозаписываем в конец ('a')
    txt_mode = 'a' if existing_results else 'w'

    iterator = tqdm(range(0, total_to_process, batch_size), desc="Processing")

    for i in iterator:
        if interrupted:
            break

        batch = image_paths_to_process[i: i + batch_size]
        batch_results = model.predict(batch, args.prompt, cleaner)

        if not batch_results:
            continue

        # Сохранение результатов
        for r in batch_results:
            results.append(asdict(r))

        # 1. Сохранение JSON
        try:
            out_path.write_text(
                json.dumps(results, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Ошибка сохранения JSON: {e}")

        # 2. Дублирование в текстовый файл
        try:
            # После первой итерации меняем режим на дозапись, чтобы не стереть предыдущие батчи текущего запуска
            # (актуально, если скрипт запущен с нуля: 1-й батч 'w', последующие 'a')
            current_txt_mode = txt_mode if i == 0 else 'a'

            with open(txt_log_path, current_txt_mode, encoding="utf-8") as tf:
                for r in batch_results:
                    # Одна строка - одно описание
                    tf.write(r.prediction + "\n")
        except Exception as e:
            logger.error(f"Ошибка записи в текстовый файл: {e}")

    logger.info(f"Готово. Обработано {len(image_paths_to_process)} файлов.")
    logger.info(f"Результаты JSON: {out_path}")
    logger.info(f"Текстовый файл описаний: {txt_log_path}")


if __name__ == "__main__":
    main()