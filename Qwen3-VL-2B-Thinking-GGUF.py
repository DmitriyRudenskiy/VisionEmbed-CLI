import os
import re
import json
import logging
import argparse
import signal
import sys
from pathlib import Path
from typing import Optional, Set, List
from dataclasses import dataclass, asdict
from PIL import Image
import torch
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
    status: str = "success"
    error: Optional[str] = None


# ─── Очистка текста ───
class TextCleaner:
    def __init__(self, lowercase: bool = False, remove_thinking: bool = True):
        self.lowercase = lowercase
        self.remove_thinking = remove_thinking
        self.allowed_pattern = re.compile(r"[^\w\s.,:;!?()\-\[\]\'\"«»„“/]", re.UNICODE)

    def clean(self, raw_text: str) -> str:
        text = raw_text
        if self.remove_thinking:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = self.allowed_pattern.sub("", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if self.lowercase:
            cleaned = cleaned.lower()
        return cleaned


# ─── Сканер (синхронный) ───
def scan_images(directory: Path, extensions: Set[str]) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory.resolve()}")

    images = [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return images


# ─── Модель (синхронный batch inference) ───
class ModelInference:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Thinking",
        max_new_tokens: int = 4096,
    ):
        logger.info(f"Loading model {model_name}...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

        self.greedy = False
        self.top_p = 0.95
        self.top_k = 20
        self.repetition_penalty = 1.0
        self.temperature = 1.0
        logger.info("Model loaded")

    def predict(self, image_paths: List[str], cleaner: TextCleaner) -> List[ProcessingResult]:
        if not image_paths:
            return []

        results: List[ProcessingResult] = []
        images: List[Image.Image] = []
        valid_paths: List[str] = []

        # Загружаем изображения
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                logger.error(f"Failed to load image {path}: {e}")
                results.append(
                    ProcessingResult(
                        image=path,
                        prediction="",
                        status="error",
                        error=f"Image load error: {e}",
                    )
                )

        if not images:
            return results

        try:
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "Create a descriptive detailed caption for this image."},
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

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=not self.greedy,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                )

            trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_texts = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for path, raw in zip(valid_paths, raw_texts):
                cleaned = cleaner.clean(raw)
                results.append(ProcessingResult(image=path, prediction=cleaned))

        except Exception as e:
            logger.error(f"Inference error: {e}")
            for path in valid_paths:
                results.append(
                    ProcessingResult(
                        image=path,
                        prediction="",
                        status="error",
                        error=f"Inference error: {e}",
                    )
                )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results


# ─── Main (синхронный конвейер) ───
def main():
    parser = argparse.ArgumentParser(description="Batch image captioning with Qwen3-VL")
    parser.add_argument(
        "directory",
        type=str,
        help="Folder with images (use '.' for current directory)",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Thinking")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--output", type=str, default=None, help="Path to results.json")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--extensions",
        default=".jpg,.jpeg,.png,.bmp,.gif,.webp,.tiff,.tif",
    )
    args = parser.parse_args()

    # ─── Определение путей ───
    target_dir = Path(args.directory).resolve()
    out_path = Path(args.output).resolve() if args.output else target_dir / "results.json"

    logger.info(f"Scanning directory: {target_dir}")

    # ─── Graceful shutdown ───
    interrupted = False

    def _handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        logger.warning("Interrupted! Saving progress...")

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    # ─── Сканирование ───
    extensions = set(args.extensions.split(","))
    try:
        image_paths = scan_images(target_dir, extensions)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    total = len(image_paths)
    logger.info(f"Found {total} image(s) in {target_dir}")

    if total == 0:
        logger.warning("No images to process. Exiting.")
        sys.exit(0)

    # ─── Инициализация ───
    cleaner = TextCleaner(lowercase=args.lowercase)
    model = ModelInference(
        model_name=args.model,
        max_new_tokens=args.max_tokens,
    )
    model.greedy = args.greedy
    model.temperature = args.temperature
    model.top_p = args.top_p
    model.top_k = args.top_k

    results: List[dict] = []
    batch_size = max(1, args.batch_size)

    # ─── Синхронная обработка ───
    for i in range(0, total, batch_size):
        if interrupted:
            break

        batch = [str(p) for p in image_paths[i : i + batch_size]]
        batch_results = model.predict(batch, cleaner)

        for r in batch_results:
            data = asdict(r)
            results.append(data)
            idx = len(results)
            status = data.get("status", "success")
            logger.info(f"[{idx}/{total}] {data['image']} → status={status}")

        # Автосохранение каждые 10 изображений
        if len(results) % 10 == 0 or i + batch_size >= total or interrupted:
            try:
                out_path.write_text(
                    json.dumps(results, indent=4, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.info(f"Checkpoint saved → {out_path}")
            except Exception as e:
                logger.error(f"Save failed: {e}")

    # ─── Финальное сохранение ───
    try:
        out_path.write_text(
            json.dumps(results, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        processed = len(results)
        successful = sum(1 for r in results if r.get("status") == "success")
        logger.info(
            f"Done. Processed {processed}/{total} images "
            f"({successful} success, {processed - successful} errors). "
            f"Results: {out_path}"
        )
    except Exception as e:
        logger.error(f"Final save failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()