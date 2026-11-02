import gradio as gr
from PIL import Image
from predict import predict


def run(img):
    if isinstance(img, dict): img = Image.fromarray(img['composite'])
    return predict(img)


gr.Interface(run, gr.Sketchpad(type='numpy'), gr.Label(num_top_classes=3), title='Распознавание цифр').launch()
