import torch
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
import warnings
import argparse
import os

# Игнорируем предупреждения о депрекации для чистоты вывода
warnings.filterwarnings("ignore", category=FutureWarning)


def format_timestamp(seconds):
    """Конвертирует секунды в формат SRT: ЧЧ:ММ:СС,мс"""
    if seconds is None:
        return "00:00:00,000"

    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3600000
    milliseconds %= 3600000
    minutes = milliseconds // 60000
    milliseconds %= 60000
    seconds = milliseconds // 1000
    milliseconds %= 1000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def main():
    # --- Настройка аргументов командной строки ---
    parser = argparse.ArgumentParser(
        description="Распознавание речи (VibeVoice-ASR-HF) с диаризацией и сохранением в SRT")
    parser.add_argument("input_file", help="Путь к исходному аудиофайлу (например, audio.mp3)")
    parser.add_argument("--prompt", default=None,
                        help="Контекст или горячие слова (hotwords) для улучшения распознавания")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="Размер чанка для генерации (если не хватает видеопамяти). По умолчанию 1440000 (60 сек)")
    args = parser.parse_args()

    audio_path = args.input_file

    # Проверка существования файла
    if not os.path.exists(audio_path):
        print(f"Ошибка: Файл '{audio_path}' не найден.")
        return

    # --- 1. Настройка устройства и типа данных ---
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float32

    # --- 2. Загрузка модели и процессора ---
    model_id = "microsoft/VibeVoice-ASR-HF"

    print(f"Загрузка модели {model_id}...")

    # Используем device_map="auto" для автоматического распределения по памяти
    processor = AutoProcessor.from_pretrained(model_id)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto"
    )

    print(f"Модель загружена на {model.device} с dtype {model.dtype}")

    # --- 3. Подготовка входных данных ---
    print(f"Подготовка аудиофайла: {audio_path}")

    inputs = processor.apply_transcription_request(
        audio=audio_path,
        prompt=args.prompt
    ).to(model.device, model.dtype)

    # --- 4. Запуск распознавания ---
    print("Начинаю распознавание (это может занять время для длинных файлов)...")

    try:
        generate_kwargs = {"**inputs": inputs}

        # Если указан кастомный chunk_size (для экономии VRAM)
        if args.chunk_size is not None:
            generate_kwargs["tokenizer_chunk_size"] = args.chunk_size

        # Генерация
        output_ids = model.generate(**inputs, tokenizer_chunk_size=args.chunk_size if args.chunk_size else 1440000)

        # Отрезаем prompt токены
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # Парсинг вывода в список словарей
        transcription = processor.decode(generated_ids, return_format="parsed")[0]

        if isinstance(transcription, str):
            print("\nОшибка: Модель вернула строку вместо распарсенного JSON. Возможно, некорректный формат вывода.")
            print("Сырой вывод:", transcription)
            return

        # --- 5. Формирование и сохранение результата в SRT ---
        file_dir = os.path.dirname(audio_path)
        file_name = os.path.basename(audio_path)

        # Меняем расширение на .srt
        name_without_ext = os.path.splitext(file_name)[0]
        output_filename = f"{name_without_ext}.srt"
        output_path = os.path.join(file_dir, output_filename) if file_dir else output_filename

        srt_content = ""
        subtitle_index = 1

        for chunk in transcription:
            text = chunk.get("Content", "").strip()
            if not text:
                continue

            start_time = chunk.get("Start", 0.0)
            end_time = chunk.get("End", 0.0)
            speaker = chunk.get("Speaker")

            # Добавляем метку говорящего
            if speaker is not None:
                text = f"[Speaker {speaker}]: {text}"

            start_str = format_timestamp(start_time)
            end_str = format_timestamp(end_time)

            # Формат SRT
            srt_content += f"{subtitle_index}\n"
            srt_content += f"{start_str} --> {end_str}\n"
            srt_content += f"{text}\n\n"

            subtitle_index += 1

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        print(f"\n--- Готово! ---")
        print(f"Субтитры с диаризацией сохранены в файл: {output_path}")

    except Exception as e:
        print(f"Произошла ошибка при обработке: {e}")


if __name__ == "__main__":
    main()