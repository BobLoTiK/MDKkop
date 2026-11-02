import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons

# UTF-8 для консоли
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

# Картинки сохраняем рядом со скриптом, а не в текущей папке.
HERE = Path(__file__).resolve().parent


def save_png(fig, name):
    path = HERE / name
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print('  картинка сохранена:', path.name)


def accuracy(p, y, thr=0.5):
    """Доля верных ответов: класс = «вероятность больше порога»."""
    return ((p > thr) == y).mean()


def plot_boundary(ax, X, y, forward, title):
    """Граница решения.

    Строим сетку 200×200 точек, для каждой считаем forward (вероятность
    класса 1) и закрашиваем плоскость по уровню 0.5. Линия, где вероятность
    ровно 0.5, и есть граница решения: по одну сторону нейрон говорит
    «класс 0», по другую — «класс 1».
    """
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
    )
    zz = forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=25)
    ax.set_title(title)
    ax.set_xlabel('признак 1')
    ax.set_ylabel('признак 2')


# Шаг 1. Два класса точек
# make_blobs генерирует два «облака»: 200 точек, у каждой 2 признака (x1, x2).
# y — метка класса, 0 или 1. Задача нейрона — разделить эти облака.
X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.2, random_state=1)
print('X.shape =', X.shape, '| y.shape =', y.shape)
print('В классах точек:', np.bincount(y))

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.title('Шаг 1: 200 точек двух классов (make_blobs)')
plt.xlabel('признак 1')
plt.ylabel('признак 2')
plt.show()

# Шаг 2. Нейрон: взвешенная сумма + сигмоида
# z = X·w + b — «сырой» выход, сигмоида сжимает его в вероятность (0, 1).
sig = lambda z: 1 / (1 + np.exp(-z))

rng = np.random.default_rng(0)
# Инициализация малыми случайными числами — раздел 1.6 конспекта: веса
# крошечные, значит z ≈ 0, p ≈ 0.5, loss ≈ log 2 и первые шаги маленькие.
# Один нейрон учится и с нулей (та же accuracy 1.0), но малый random — это
# привычка на следующую лабу: в скрытом слое нули дают всем нейронам
# одинаковый градиент, и слой перестаёт делить признаки между собой.
w = rng.normal(0, 0.1, 2)
b = 0.0


def forward(X):
    return sig(X @ w + b)


z = X @ w + b
p = forward(X)
print()
print('Формы: w', w.shape, '| X', X.shape, '| z', z.shape, '| p', p.shape)
print('w =', w, '| b =', b)
print('accuracy до обучения:', accuracy(p, y))

# Шаг 3. Функция потерь: бинарная кросс-энтропия (BCE)
# p — предсказанная вероятность класса 1. Если y = 1, штраф = −log p,
# если y = 0, штраф = −log(1 − p). 1e-9 защищает log от нуля.
def bce(p, y):
    return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))


print('BCE до обучения:', bce(p, y), ' (≈ log 2 = 0.693: «модель ничего не знает»)')

# Шаг 4. Градиент и обучение
# Великая формула: dz = p − y, дальше просто усредняем по точкам.
#   dw = Xᵀ·dz / n,   db = mean(dz)
# 500 шагов спуска с lr = 0.1.
lr, hist = 0.1, []

for i in range(500):
    p = forward(X)
    dz = p - y
    w -= lr * X.T @ dz / len(X)
    b -= lr * dz.mean()
    hist.append(bce(p, y))

print()
print('После обучения: w =', w, '| b =', b)
print('BCE:', hist[0], '→', hist[-1])
print('accuracy после обучения:', accuracy(forward(X), y))

plt.figure(figsize=(6, 4))
plt.plot(hist)
plt.title(f'Шаг 4: потеря BCE по итерациям (accuracy {accuracy(forward(X), y):.3f})')
plt.xlabel('итерация')
plt.ylabel('BCE')
plt.show()

# Шаг 5. Граница решения
# Один нейрон умеет рисовать только прямую: z = w1·x1 + w2·x2 + b = 0 —
# это уравнение прямой, поэтому граница всегда линейная.
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)
plot_boundary(ax, X, y, forward, f'Шаг 5: граница решения, accuracy {accuracy(forward(X), y):.3f}')
save_png(fig, 'boundary_blobs.png')
plt.show()

# Шаг 6. Где нейрон бессилен
# make_moons — две «луны», вложенные друг в друга. Данные изогнуты, а граница
# у одного нейрона прямая, поэтому часть точек неизбежно попадает не туда.
Xm, ym = make_moons(n_samples=200, noise=0.2, random_state=0)

# Тот же нейрон, обученный заново с нуля.
rng = np.random.default_rng(0)
wm = rng.normal(0, 0.1, 2)
bm = 0.0


def forward_m(X):
    return sig(X @ wm + bm)


lr_m, hist_m = 0.1, []
for i in range(500):
    pm = forward_m(Xm)
    dz = pm - ym
    wm -= lr_m * Xm.T @ dz / len(Xm)
    bm -= lr_m * dz.mean()
    hist_m.append(bce(pm, ym))

print()
print('Луны: w =', wm, '| b =', bm)
print('accuracy на make_moons:', accuracy(forward_m(Xm), ym), '(около 0.85 — предел прямой)')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(hist_m)
ax1.set_title('Шаг 6: потеря на «лунах»')
ax1.set_xlabel('итерация')
ax1.set_ylabel('BCE')
plot_boundary(ax2, Xm, ym, forward_m, f'Шаг 6: граница всё ещё прямая, accuracy {accuracy(forward_m(Xm), ym):.3f}')
save_png(fig, 'boundary_moons.png')
plt.show()

# Шаг 7. Домашнее повторение: tanh вместо сигмоиды
# tanh(z) даёт значения (−1, 1), порог класса — 0 (а не 0.5). Вероятность
# получаем сдвигом: p = (tanh(z) + 1) / 2. Градиент — по цепному правилу,
# как в задании, через производную tanh: dt/dz = 1 − t²:
#   dL/dz = dL/dp · dp/dt · dt/dz = (p − y)/(p(1 − p)) · 1/2 · (1 − t²).
# Полезное тождество: tanh(z) = 2·σ(2z) − 1, то есть p = σ(2z) — тот же
# логистический нейрон с удвоенным z. Алгебра сокращает формулу до 2·(p − y);
# ниже печатаем, что оба способа дают один и тот же градиент.
rng = np.random.default_rng(0)
wt = rng.normal(0, 0.1, 2)
bt = 0.0


def forward_t(X):
    return (np.tanh(X @ wt + bt) + 1) / 2


lr_t, hist_t, grad_diff = 0.1, [], 0.0
for i in range(500):
    # t обрезаем на 1e-9: при |z| ≳ 19 tanh округляется ровно до ±1,
    # и формула ниже делила бы на p(1 − p) = 0.
    t = np.clip(np.tanh(X @ wt + bt), -1 + 1e-9, 1 - 1e-9)
    pt = (t + 1) / 2
    dz = (pt - y) / (pt * (1 - pt)) * 0.5 * (1 - t ** 2)
    grad_diff = max(grad_diff, np.abs(dz - 2 * (pt - y)).max())
    wt -= lr_t * X.T @ dz / len(X)
    bt -= lr_t * dz.mean()
    hist_t.append(bce(pt, y))

# Проверяем, что порог 0 даёт те же предсказания: tanh(z) > 0 ⇔ p > 0.5.
pred_t = (np.tanh(X @ wt + bt) > 0).astype(int)
print()
print('Домашка (tanh): w =', wt, '| b =', bt)
print('BCE:', hist_t[0], '→', hist_t[-1])
print('accuracy с порогом 0:', (pred_t == y).mean())
print('градиент: цепное правило (1 − t²) vs сокращённое 2(p − y), макс. разница =', grad_diff)

# Прямая проверка тождества: tanh-нейрон с половинными весами — это ровно
# сигмоида. Разница должна быть на уровне погрешности float64 (~1e-16).
p_half = (np.tanh(X @ (w / 2) + b / 2) + 1) / 2
print('max |p_сигмоида − p_tanh(w/2)| =', np.abs(forward(X) - p_half).max())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plot_boundary(ax1, X, y, forward, f'Сигмоида, порог 0.5 — accuracy {accuracy(forward(X), y):.3f}')
plot_boundary(ax2, X, y, forward_t, f'tanh, порог 0 — accuracy {accuracy(pred_t, y):.3f}')
fig.suptitle('Шаг 7 (домашка): сигмоида и tanh дают одну и ту же прямую')
plt.tight_layout()
save_png(fig, 'boundary_tanh.png')
plt.show()
