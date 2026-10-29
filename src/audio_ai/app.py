import csv

import gradio as gr

from audio_ai.assistant import run_omni_assistant


def answer(image, audio):
    if image is None or audio is None:
        return "Загрузите изображение и аудио с вопросом.", None
    text, audio_path, _, _ = run_omni_assistant(image, audio)
    return text, audio_path


def load_examples():
    with open("data/examples.csv", encoding="utf-8") as file:
        return [["data/" + row["image"], "data/" + row["audio"]] for row in csv.DictReader(file)]


with gr.Blocks(title="Голосовой ассистент") as demo:
    gr.Markdown("# Голосовой ассистент\nЗагрузите картинку и задайте вопрос голосом.")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Изображение", type="filepath")
            audio_input = gr.Audio(
                label="Вопрос", sources=["microphone", "upload"], type="filepath"
            )
            button = gr.Button("Спросить", variant="primary")
        with gr.Column():
            text_output = gr.Textbox(label="Ответ ассистента", lines=5)
            audio_output = gr.Audio(label="Голосовой ответ", autoplay=True)
    gr.Examples(load_examples(), inputs=[image_input, audio_input])
    button.click(answer, inputs=[image_input, audio_input], outputs=[text_output, audio_output])


if __name__ == "__main__":
    demo.launch()
