import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

# UTF-8 для консоли
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
torch.set_num_threads(4)  # число ядер процессора

# Шаг 1. Скачать и посмотреть — 15 мин
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train = datasets.MNIST(str(DATA), train=True, download=True, transform=tf)
test = datasets.MNIST(str(DATA), train=False, download=True, transform=tf)
print(len(train), 'train,', len(test), 'test')
img, label = train[0]; print(img.shape, label)
fig, ax = plt.subplots(5, 5, figsize=(6, 6))
for i, a in enumerate(ax.flat):
    a.imshow(train[i][0][0], cmap='gray'); a.set_title(train[i][1]); a.axis('off')
fig.suptitle('первые 25 картинок MNIST')
plt.savefig(HERE / 'mnist_grid.png', bbox_inches='tight'); plt.show()

# Шаг 2. DataLoader — 10 мин
train_dl = DataLoader(train, batch_size=128, shuffle=True)
test_dl = DataLoader(test, batch_size=512)
x, y = next(iter(train_dl)); print(x.shape, y.shape, len(train_dl))

# Шаг 3. Модель — 10 мин
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

# Шаг 4. Цикл по эпохам — 25 мин
for ep in range(3):
    t0 = time.time(); model.train()
    for x, y in train_dl:
        opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in test_dl: correct += (model(x).argmax(1) == y).sum().item()
    sched.step()
    print(f'epoch {ep} loss {loss.item():.3f} acc {correct / len(test):.4f} lr {sched.get_last_lr()[0]} time {time.time()-t0:.0f}s')

# Шаг 5. Ошибки модели — 15 мин
model.eval()
preds, trues = [], []
with torch.no_grad():
    for x, y in test_dl:
        preds.append(model(x).argmax(1)); trues.append(y)
preds = torch.cat(preds); trues = torch.cat(trues)
wrong = (preds != trues).nonzero().flatten()
print('ошибок на тесте:', len(wrong), 'из', len(test))
fig, ax = plt.subplots(4, 4, figsize=(7, 8))
for i, a in enumerate(ax.flat):
    j = wrong[i].item()
    a.imshow(test[j][0][0], cmap='gray'); a.set_title(f'{preds[j].item()} / {trues[j].item()}'); a.axis('off')
fig.suptitle('16 ошибок сети: в подписи сверху предсказано, снизу правда')
plt.savefig(HERE / 'errors.png', bbox_inches='tight'); plt.show()

cm = confusion_matrix(trues.numpy(), preds.numpy())
print(cm)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues'); ax.set_xlabel('предсказано'); ax.set_ylabel('правда')
ax.set_xticks(range(10)); ax.set_yticks(range(10))
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7, color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.colorbar(im); plt.title('матрица ошибок: где сеть путает цифры'); plt.savefig(HERE / 'confusion.png', bbox_inches='tight'); plt.show()
print('самые частые путаницы (не диагональ):')
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 10:
            print(f'  {i} → {j}: {cm[i, j]}')

# Шаг 6. Сохранить и загрузить — 10 мин
torch.save(model.state_dict(), HERE / 'mnist.pt')
m2 = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
m2.load_state_dict(torch.load(HERE / 'mnist.pt')); m2.eval()
correct2 = 0
with torch.no_grad():
    for x, y in test_dl: correct2 += (m2(x).argmax(1) == y).sum().item()
print('загруженная модель: acc', correct2 / len(test), '| размер файла', (HERE / 'mnist.pt').stat().st_size, 'байт')

# Шаг 7. Эксперимент — 5 мин
def run(lr=1e-3, normalize=True, epochs=3, hidden=128):
    t = transforms.Compose([transforms.ToTensor()] + ([transforms.Normalize((0.1307,), (0.3081,))] if normalize else []))
    tr = datasets.MNIST(str(DATA), train=True, transform=t)
    te = datasets.MNIST(str(DATA), train=False, transform=t)
    dl = DataLoader(tr, batch_size=128, shuffle=True); tl = DataLoader(te, batch_size=512)
    m = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, 10))
    o = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        m.train()
        for x, y in dl:
            o.zero_grad(); loss_fn(m(x), y).backward(); o.step()
    m.eval(); c = 0
    with torch.no_grad():
        for x, y in tl: c += (m(x).argmax(1) == y).sum().item()
    return c / len(te)

print()
print('эксперимент: lr 1e-3 + Normalize (основной) acc', round(correct2 / len(test), 4))
print('эксперимент: lr 0.1 + Normalize              acc', round(run(lr=0.1), 4))
print('эксперимент: lr 1e-3 без Normalize           acc', round(run(normalize=False), 4))
# Дополнительно: при 128 нейронах и 3 эпохах выходит ~0.968, то есть чуть ниже
# порога 0.97 из «Результата занятия». Задание само подсказывает выход —
# 256 нейронов («128 вместо 256 — точность почти та же») или четвёртая эпоха.
print('эксперимент: 256 нейронов, 3 эпохи           acc', round(run(hidden=256), 4))
print('эксперимент: 128 нейронов, 4 эпохи           acc', round(run(epochs=4), 4))
