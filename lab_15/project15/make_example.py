"""Картинка-пример для README: цифра из MNIST и три ответа сети."""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image

from predict import predict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
test = datasets.MNIST(str(HERE / 'data'), train=False, download=True, transform=transforms.ToTensor())

fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
for ax, idx in zip(axes, [0, 7, 15]):
    img, label = test[idx]
    pil = Image.fromarray((img[0].numpy() * 255).astype(np.uint8))
    probs = predict(pil)
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
    ax.imshow(pil, cmap='gray')
    ax.set_title(f'правда: {label}\n' + '\n'.join(f'{d}: {p:.2f}' for d, p in top))
    ax.axis('off')
plt.tight_layout()
plt.savefig(HERE / 'example.png', bbox_inches='tight')
print('example.png сохранён')
