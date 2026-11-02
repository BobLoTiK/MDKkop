import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons

# UTF-8 для консоли
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

# Картинки границ — артефакт занятия, кладём рядом со скриптом.
HERE = Path(__file__).resolve().parent

# Шаг 1. Два класса точек — 10 мин
X, y = make_blobs(200, centers=2, cluster_std=1.2, random_state=1)
plt.figure(); plt.title('данные: два облака точек'); plt.scatter(X[:, 0], X[:, 1], c=y); plt.show()

# Шаг 2. Нейрон — 15 мин
sig = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(0)
w = rng.normal(0, 0.1, 2); b = 0.0
def forward(X): return sig(X @ w + b)
print(((forward(X) > .5) == y).mean())

# Шаг 3. Loss: BCE — 15 мин
def bce(p, y): return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
print(bce(forward(X), y))

# Шаг 4. Градиент и обучение — 20 мин
lr, hist = 0.1, []
for i in range(500):
    p = forward(X); dz = p - y
    w -= lr * X.T @ dz / len(X); b -= lr * dz.mean()
    hist.append(bce(p, y))
plt.figure(); plt.title('функция потерь BCE по шагам обучения'); plt.plot(hist); plt.show()
print('accuracy:', ((forward(X) > .5) == y).mean())

# Шаг 5. Граница решения — 15 мин
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200), np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
zz = forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure()
plt.title('граница решения на двух облаках: прямая справляется')
plt.contourf(xx, yy, zz, levels=[0, .5, 1], alpha=.3); plt.scatter(X[:,0], X[:,1], c=y)
plt.savefig(HERE / 'boundary_blobs.png', bbox_inches='tight'); plt.show()

# Шаг 6. Где нейрон бессилен — 15 мин
# Параметров make_moons в задании нет; эти дают обещанные «около 0.85».
X, y = make_moons(200, noise=0.2, random_state=0)
rng = np.random.default_rng(0)
w = rng.normal(0, 0.1, 2); b = 0.0
for i in range(500):
    p = forward(X); dz = p - y
    w -= lr * X.T @ dz / len(X); b -= lr * dz.mean()
print('accuracy на moons:', ((forward(X) > .5) == y).mean())
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200), np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
zz = forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure()
plt.title('граница решения на лунах: одна прямая не справляется')
plt.contourf(xx, yy, zz, levels=[0, .5, 1], alpha=.3); plt.scatter(X[:,0], X[:,1], c=y)
plt.savefig(HERE / 'boundary_moons.png', bbox_inches='tight'); plt.show()

# Домашнее повторение: сигмоиду на tanh, порог 0, производная 1 − tanh²
X, y = make_blobs(200, centers=2, cluster_std=1.2, random_state=1)
rng = np.random.default_rng(0)
w = rng.normal(0, 0.1, 2); b = 0.0
for i in range(500):
    t = np.tanh(X @ w + b)
    p = (t + 1) / 2
    dz = (p - y) / (p * (1 - p) + 1e-9) * 0.5 * (1 - t ** 2)
    w -= 0.1 * X.T @ dz / len(X); b -= 0.1 * dz.mean()
print('accuracy на tanh (порог 0):', ((np.tanh(X @ w + b) > 0) == y).mean())
