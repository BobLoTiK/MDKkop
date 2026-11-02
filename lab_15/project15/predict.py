from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
model.load_state_dict(torch.load(HERE / 'mnist.pt')); model.eval()


def predict(img):
    img = ImageOps.grayscale(img).resize((28, 28))
    x = np.array(img, dtype=np.float32) / 255
    if x.mean() > 0.5: x = 1 - x          # инверсия при светлом фоне
    x = (x - 0.1307) / 0.3081
    with torch.no_grad(): p = model(torch.tensor(x)[None, None]).softmax(1)[0]
    return {str(i): float(p[i]) for i in range(10)}
