import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader

# UTF-8 для консоли
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data' / 'hymenoptera_data'
PHOTOS = [Path(p) for p in sys.argv[1:]]
torch.manual_seed(0)

# Шаг 1. Данные — 10 мин
# train/ants, train/bees, val/ants, val/bees; Resize 256, CenterCrop 224, нормировка ImageNet.
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
tr = datasets.ImageFolder(str(DATA / 'train'), tf); va = datasets.ImageFolder(str(DATA / 'val'), tf)
print(tr.classes, len(tr), len(va))

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
for i, ax in enumerate(axes.flat):
    img, label = tr[i]
    ax.imshow((img * std + mean).clamp(0, 1).permute(1, 2, 0))
    ax.set_title(tr.classes[label]); ax.axis('off')
plt.tight_layout(); plt.savefig(HERE / 'dataset.png', bbox_inches='tight'); plt.show()

# Шаг 2. С нуля: маленькая CNN на 96×96 — 15 мин
tf96 = T.Compose([T.Resize(96), T.CenterCrop(96), T.ToTensor(), norm])
tr96 = datasets.ImageFolder(str(DATA / 'train'), tf96)
va96 = datasets.ImageFolder(str(DATA / 'val'), tf96)
dl96 = DataLoader(tr96, batch_size=32, shuffle=True); vl96 = DataLoader(va96, batch_size=64)
cnn = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(32 * 24 * 24, 2),
)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
t0 = time.perf_counter()
for ep in range(5):
    cnn.train()
    for x, y in dl96:
        opt.zero_grad(); loss_fn(cnn(x), y).backward(); opt.step()
    cnn.eval(); c = n = 0
    with torch.no_grad():
        for x, y in vl96:
            c += (cnn(x).argmax(1) == y).sum().item(); n += len(y)
    print(f'CNN с нуля, эпоха {ep}: val acc {c / n:.3f}')
cnn_time = time.perf_counter() - t0
cnn_acc = c / n

# Шаг 3. Признаки один раз — 20 мин
body = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT); body.classifier = nn.Identity(); body.eval()


def extract(ds):
    xs, ys = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, 32):
            xs.append(body(x)); ys.append(y)
    return torch.cat(xs), torch.cat(ys)


t0 = time.perf_counter()
Ftr, ytr = extract(tr); Fva, yva = extract(va)
extract_time = time.perf_counter() - t0
print(Ftr.shape, Fva.shape, 'признаки считаны за', round(extract_time, 1), 'с')
torch.save((Ftr, ytr, Fva, yva), HERE / 'features.pt')

# Шаг 4. Голова на признаках — 15 мин
head = nn.Linear(576, 2); opt = torch.optim.Adam(head.parameters(), 1e-3); loss_fn = nn.CrossEntropyLoss()
t0 = time.perf_counter()
for ep in range(100):
    opt.zero_grad(); loss_fn(head(Ftr), ytr).backward(); opt.step()
head_time = time.perf_counter() - t0
head_acc = (head(Fva).argmax(1) == yva).float().mean().item()
print('val acc:', head_acc)
print(sum(p.numel() for p in head.parameters()), 'обучаемых параметров')

# Шаг 5. Собрать модель целиком — 10 мин
body.classifier = head; body.eval()
img, label = va[0]
print('предсказание на val[0]:', body(img[None]).argmax(1).item(), '| правда:', label)
torch.save(body.state_dict(), HERE / 'ants_bees.pt')

# Шаг 6. Своё фото — 10 мин
# Фото пчелы и муравья; фото кота даст «пчелу» или «муравья» — модель знает два класса.
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
for path in PHOTOS:
    if not path.exists():
        print('нет файла', path)
        continue
    from PIL import Image
    x = tf(Image.open(path).convert('RGB'))
    with torch.no_grad():
        prob = body(x[None]).softmax(1)[0]
    print(path.name, '→', ', '.join(f'{tr.classes[i]} {v:.2f}' for i, v in enumerate(prob)))

# Шаг 7. Мостик к LoRA — 10 мин
# Обучили 0.05 % параметров MobileNet (1154 из 2.5 млн) и получили результат.
print()
print('обучаемых параметров головы:', sum(p.numel() for p in head.parameters()),
      'из', sum(p.numel() for p in body.parameters()), 'всей сети')

print()
print('таблица:')
print(f'{"вариант":<28} {"val acc":>8} {"время, с":>10}')
print(f'{"CNN с нуля (96×96, 5 эпох)":<28} {cnn_acc:>8.3f} {cnn_time:>10.1f}')
print(f'{"MobileNet признаки + голова":<28} {head_acc:>8.3f} {extract_time + head_time:>10.1f}')
