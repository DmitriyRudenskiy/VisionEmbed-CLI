import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import warnings
import argparse
import os

# Игнорируем предупреждения о депрекации для чистоты вывода
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    # --- Настройка аргументов командной строки ---
    parser = argparse.ArgumentParser(description="Распознавание речи из аудиофайла (Whisper Large V3)")
    parser.add_argument("input_file", help="Путь к исходному аудиофайлу (например, audio.mp3)")
    args = parser.parse_args()

    audio_path = args.input_file

    # Проверка существования файла
    if not os.path.exists(audio_path):
        print(f"Ошибка: Файл '{audio_path}' не найден.")
        return

    # --- 1. Настройка устройства ---
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32  # float32 стабильнее на MPS
    elif torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    print(f"Устройство: {device}, тип данных: {torch_dtype}")

    # --- 2. Загрузка модели с Hugging Face ---
    # Используем официальный идентификатор модели.
    # При первом запуске модель скачается автоматически (около 3 Гб).
    model_id = "openai/whisper-large-v3"

    print(f"Загрузка модели {model_id} (это может занять время при первом запуске)...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # --- 3. Создание пайплайна ---
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,  # Разбивает длинный файл на кусочки
    )

    # --- 4. Запуск распознавания ---
    print(f"Начинаю обработку файла: {audio_path}")

    try:
        # return_timestamps=True нужен для файлов длиннее 30 сек
        result = pipe(audio_path, return_timestamps=True)
        recognized_text = result["text"]

        # --- 5. Сохранение результата ---
        # Получаем директорию и имя исходного файла
        file_dir = os.path.dirname(audio_path)
        file_name = os.path.basename(audio_path)

        # Меняем расширение на .txt
        name_without_ext = os.path.splitext(file_name)[0]
        output_filename = f"{name_without_ext}.txt"

        # Формируем полный путь к новому файлу
        output_path = os.path.join(file_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(recognized_text)

        print("\n--- Готово! ---")
        print(f"Текст успешно сохранен в файл: {output_path}")

    except Exception as e:
        print(f"Произошла ошибка при обработке: {e}")


if __name__ == "__main__":
    main()