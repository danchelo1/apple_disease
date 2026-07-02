import gradio as gr

from app.model import LABELS_RU, gradcam, load_model, predict

model = load_model()  # MODEL_NAME из env, по умолчанию 'resnet'


def analyze(image):
    if image is None:
        return {}, None
    probs = predict(model, image)
    labeled = {LABELS_RU.get(cls, cls): p for cls, p in probs.items()}
    overlay, _ = gradcam(model, image)
    return labeled, overlay


with gr.Blocks(title="Apple Leaf Disease") as demo:
    gr.Markdown(
        "# 🍏 Классификация болезней листьев яблони\n"
        "Загрузите фото листа — модель определит болезнь и покажет **Grad-CAM** "
        "(на какую область смотрит сеть)."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Фото листа")
            btn = gr.Button("Анализировать", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=4, label="Диагноз")
            out_cam = gr.Image(label="Grad-CAM (зоны внимания сети)")

    btn.click(analyze, inputs=inp, outputs=[out_label, out_cam])
    inp.upload(analyze, inputs=inp, outputs=[out_label, out_cam])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
