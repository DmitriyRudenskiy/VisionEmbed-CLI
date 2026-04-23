from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# === Настройки генерации ===
greedy = False
top_p = 0.95
top_k = 20
repetition_penalty = 1.0
presence_penalty = 0.0
temperature = 1.0
out_seq_length = 40960

# === Путь к локальному изображению ===
local_image_path = "/Users/user/Downloads/Ain Nguyen – Ouro Kronii Lingerie – CosBlay_files/Ain-Nguyen-Ouro-Kronii-Lingerie-4.jpg"

# Загрузка модели (БЕЗ GGUF!)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Thinking",  # <-- исправлено!
    torch_dtype="auto",            # <-- dtype -> torch_dtype
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Thinking")  # <-- исправлено!

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": local_image_path,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference
generated_ids = model.generate(
    **inputs,
    max_new_tokens=out_seq_length,
    do_sample=not greedy,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    repetition_penalty=repetition_penalty,
    presence_penalty=presence_penalty
)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)