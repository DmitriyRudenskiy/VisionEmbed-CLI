# 📸 VisionEmbed-CLI

Консольная утилита для векторизации изображений с использованием VL-моделей (Vision-Language).
Автоматически обрабатывает папки, оптимизирует размер изображений и сохраняет результат в JSON.

## 🚀 Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/ваш_юзернейм/VisionEmbed-CLI.git
   cd VisionEmbed-CLI
   ```

2. Создайте виртуальное окружение (рекомендуется):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## ⬇️ Скачивание модели

Вы можете запустить скрипт без предзагрузки модели (она скачается автоматически при первом запуске), либо скачать модель заранее:

```bash
# Установка утилиты для скачивания
pip install huggingface_hub

# Скачивание модели в локальную папку ./model
huggingface-cli download Qwen/Qwen3-VL-Embedding-2B --local-dir ./model
```

## 🛠️ Использование

**Базовый запуск:**
```bash
python vectorize.py --input_dir ./путь_к_фото --output result.json
```

**Использование локальной модели и изменение размера:**
```bash
python vectorize.py --input_dir ./photos --model_path ./model --resize 384 --output data.json
```

### Аргументы командной строки:

| Аргумент | Описание | По умолчанию |
| :--- | :--- | :--- |
| `--input_dir` | **(Обязательно)** Путь к папке с изображениями | - |
| `--output` | Имя выходного JSON файла | `embeddings.json` |
| `--model_path` | ID модели на HF или путь к локальной папке | `Qwen/Qwen3-VL-Embedding-2B` |
| `--resize` | Макс. размер стороны (пикс). 0 — не менять. | `384` |

## 📄 Формат вывода (JSON)

Выходной файл представляет собой массив объектов:

```json
[
    {
        "image_path": "photos/cat.jpg",
        "embedding_dim": 1024,
        "embedding": [0.123, -0.543, ...]
    }
]
```