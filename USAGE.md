# Использование моделей с llama.cpp

## Обзор

Данное руководство описывает, как запускать мультимодальные модели (в частности, Qwen3-VL) с помощью **llama.cpp** через командную строку или веб-интерфейс.

---

## Требования

- Установленный **llama.cpp** (последняя версия)
- GGUF-файлы модели и проектора (mmproj)
- Изображение для анализа (при использовании VLM)

---

## Установка

Убедитесь, что вы используете актуальную версию llama.cpp:

- **Сборка из исходников:** клонируйте репозиторий и соберите проект согласно [официальной инструкции](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)
- **Готовый релиз:** скачайте последний билд для вашей платформы из раздела [Releases](https://github.com/ggerganov/llama.cpp/releases)

---

## Способы запуска

### 1. CLI Inference (`llama-mtmd-cli`)

Подходит для быстрого тестирования и скриптовой обработки изображений.

#### Пример: Qwen3-VL-2B-Instruct

```bash
llama-mtmd-cli \
  -m path/to/Qwen3VL-2B-Thinking-Q8_0.gguf \
  --mmproj path/to/mmproj-Qwen3VL-2B-Thinking-F16.gguf \
  --image test.jpeg \
  -p "What is the publisher name of the newspaper?" \
  --temp 1.0 --top-k 20 --top-p 0.95 -n 1024
```

#### Описание параметров

| Параметр | Описание |
|----------|----------|
| `-m` | Путь к GGUF-файлу языковой модели |
| `--mmproj` | Путь к GGUF-файлу проектора (vision encoder) |
| `--image` | Путь к изображению для анализа |
| `-p` / `--prompt` | Текст запроса (prompt) |
| `--temp` | Температура сэмплирования (креативность ответов) |
| `--top-k` | Ограничение выборки по топ-K токенам |
| `--top-p` | Nucleus sampling (top-p) |
| `-n` | Максимальное количество генерируемых токенов |

---

### 2. Web Chat (`llama-server`)

Запускает OpenAI-compatible API с встроенным веб-чатом.

#### Пример: Qwen3-VL-235B-A22B-Instruct

```bash
llama-server \
  -m path/to/Qwen3VL-235B-A22B-Instruct-Q4_K_M-split-00001-of-00003.gguf \
  --mmproj path/to/mmproj-Qwen3VL-235B-A22B-Instruct-Q8_0.gguf
```

#### Использование

После запуска сервера:

- **Веб-интерфейс:** откройте в браузере http://localhost:8080
- **API:** отправляйте запросы на эндпоинт `/v1/chat/completions` в формате, совместимом с OpenAI API

> 💡 **Совет:** Для моделей, разбитых на несколько GGUF-файлов (шардов), укажите только первый файл (например, `...-00001-of-00003.gguf`). llama.cpp автоматически подгрузит все остальные части.