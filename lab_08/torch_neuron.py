import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_blobs, make_moons

# UTF-8 для консоли
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent

# Данные из задачи 7: те же два облака, 200 точек, 2 признака.
X, y = make_blobs(200, centers=2, cluster_std=1.2, random_state=1)

# Шаг 1. Тензор и autograd — 15 мин
Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
w = torch.zeros(2, requires_grad=True); b = torch.zeros(1, requires_grad=True)
p = torch.sigmoid(Xt @ w + b)
loss = torch.nn.functional.binary_cross_entropy(p, yt)
loss.backward(); print(w.grad, b.grad)

# Сверка с ручным градиентом задачи 7 при тех же w: dw = Xᵀ·(p − y)/n, db = mean(p − y).
dz = (p - yt).detach().numpy()
print('руками:', X.T @ dz / len(X), dz.mean())

# Шаг 2. Цикл руками на тензорах — 15 мин
w = torch.zeros(2, requires_grad=True); b = torch.zeros(1, requires_grad=True)
for i in range(500):
    p = torch.sigmoid(Xt @ w + b)
    loss = torch.nn.functional.binary_cross_entropy(p, yt)
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad; b -= 0.1 * b.grad
    w.grad.zero_(); b.grad.zero_()
print(loss.item())

# Шаг 3. nn.Linear и оптимизатор — 20 мин
model = nn.Linear(2, 1)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for i in range(500):
    opt.zero_grad()
    loss = loss_fn(model(Xt).squeeze(1), yt)
    loss.backward()
    opt.step()
print(loss.item(), model.weight, model.bias)

# Шаг 4. SGD против Adam — 20 мин
def train(make_opt, steps=300):
    m = nn.Linear(2, 1); opt = make_opt(m.parameters()); h = []
    for i in range(steps):
        opt.zero_grad(); l = loss_fn(m(Xt).squeeze(1), yt); l.backward(); opt.step(); h.append(l.item())
    return h

plt.plot(train(lambda p: torch.optim.SGD(p, lr=0.1)), label='SGD')
plt.plot(train(lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)), label='SGD+momentum')
plt.plot(train(lambda p: torch.optim.Adam(p, lr=0.01)), label='Adam')
plt.legend(); plt.savefig(HERE / 'optimizers.png', bbox_inches='tight'); plt.show()

# Шаг 5. Скрытый слой — 15 мин
# Данные для всех нейросетей ниже: луны из задачи 7.
Xm, ym = make_moons(300, noise=.2, random_state=0)
Xt = torch.tensor(Xm, dtype=torch.float32); yt = torch.tensor(ym, dtype=torch.float32)
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for i in range(1000):
    opt.zero_grad(); l = loss_fn(model(Xt).squeeze(1), yt); l.backward(); opt.step()
print('accuracy:', ((model(Xt).squeeze(1) > 0).float() == yt).float().mean().item())

# Граница решения: сетка точек прогоняется через сеть, закрашиваем по уровню 0.
xx, yy = np.meshgrid(np.linspace(Xm[:,0].min()-1, Xm[:,0].max()+1, 200), np.linspace(Xm[:,1].min()-1, Xm[:,1].max()+1, 200))
with torch.no_grad():
    zz = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).reshape(xx.shape)
plt.contourf(xx, yy, zz, levels=[-1e9, 0, 1e9], alpha=.3); plt.scatter(Xm[:,0], Xm[:,1], c=ym, edgecolor='k')
plt.title('16 нейронов: граница изогнутая'); plt.savefig(HERE / 'boundary_moons.png', bbox_inches='tight'); plt.show()

# Домашнее повторение: скрытый слой 2, 4, 64 нейрона — accuracy и границы.
sizes = [2, 4, 64]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, h in zip(axes, sizes):
    net = nn.Sequential(nn.Linear(2, h), nn.ReLU(), nn.Linear(h, 1))
    o = torch.optim.Adam(net.parameters(), lr=0.01)
    for i in range(1000):
        o.zero_grad(); l = loss_fn(net(Xt).squeeze(1), yt); l.backward(); o.step()
    acc = ((net(Xt).squeeze(1) > 0).float() == yt).float().mean().item()
    with torch.no_grad():
        zz = net(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-1e9, 0, 1e9], alpha=.3); ax.scatter(Xm[:,0], Xm[:,1], c=ym, edgecolor='k')
    ax.set_title(f'{h} нейронов, accuracy {acc:.3f}')
    print(f'{h} нейронов: accuracy {acc:.3f}')
plt.tight_layout(); plt.savefig(HERE / 'hidden_sizes.png', bbox_inches='tight'); plt.show()
