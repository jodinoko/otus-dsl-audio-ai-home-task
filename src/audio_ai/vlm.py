import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MIN_PIXELS = 256 * 256
MAX_PIXELS = 1024 * 1024

SYSTEM_PROMPT = (
    "Ты голосовой ассистент. Пользователь показывает картинку и задаёт вопрос голосом. "
    "Отвечай по-русски, коротко: не больше 2-3 предложений. "
    "Твой ответ будет озвучен, поэтому не используй списки, markdown, эмодзи и цифры — "
    "пиши числа словами."
)

_model = None
_processor = None


def load_qwen():
    global _model, _processor
    if _model is None:
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            device_map={"": DEVICE},
        )
        _processor = AutoProcessor.from_pretrained(
            MODEL_NAME, size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
        )
    return _model, _processor


def ask(image_path, question, max_new_tokens=128):
    model, processor = load_qwen()
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = output[0][inputs["input_ids"].shape[1]:]
    return processor.decode(answer, skip_special_tokens=True).strip()
