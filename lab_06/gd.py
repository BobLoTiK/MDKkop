import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# UTF-8 для консоли
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

# ============================================================================
# Шаг 1–2. Спуск по параболе f(x) = (x − 3)²
# ============================================================================
# Производная f'(x) = 2(x − 3) показывает направление роста.
# Идём против градиента: x = x − lr · f'(x).
f = lambda x: (x - 3) ** 2
df = lambda x: 2 * (x - 3)

x, lr, path = 10.0, 0.1, []
for i in range(30):
    path.append(x)
    x = x - lr * df(x)

print('Итоговый x:', x)

# График параболы с траекторией спуска.
xs = np.linspace(-2, 12, 100)
plt.plot(xs, f(xs))
plt.scatter(path, [f(p) for p in path], c='r')
plt.show()


# ============================================================================
# Шаг 3. Три learning rate
# ============================================================================
# Показываем, как lr влияет на сходимость:
# lr = 0.01 — медленно, может не дойти;
# lr = 0.1 — нормально;
# lr = 1.1 — слишком много, расходится.
lrs = [0.01, 0.1, 1.1]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, cur_lr in zip(axes, lrs):
    x_cur = 10.0
    path_cur = []
    for i in range(30):
        path_cur.append(x_cur)
        x_cur = x_cur - cur_lr * df(x_cur)
    ax.plot(xs, f(xs))
    ax.scatter(path_cur, [f(p) for p in path_cur], c='r', s=20)
    ax.set_title(f'lr = {cur_lr}')

plt.tight_layout()
plt.show()


# ============================================================================
# Шаг 4. Данные для прямой
# ============================================================================
# Генерируем 50 точек: y = 2x + 1 + шум.
# Задача: найти w и b, не зная их истинных значений.
rng = np.random.default_rng(0)
X = rng.uniform(0, 5, 50)
y = 2 * X + 1 + rng.normal(0, 0.5, 50)

plt.scatter(X, y)
plt.show()


# ============================================================================
# Шаг 5. Loss и градиент
# ============================================================================
# Предсказание: p = w·X + b
# Функция потерь: MSE = mean((p − y)²)
# Градиенты: dw = mean(2(p − y)·X), db = mean(2(p − y))
def loss(w, b):
    return np.mean((w * X + b - y) ** 2)


def grads(w, b):
    p = w * X + b
    return np.mean(2 * (p - y) * X), np.mean(2 * (p - y))


# ============================================================================
# Шаг 6. Обучение прямой
# ============================================================================
# Стартуем с w = 0, b = 0, делаем 200 шагов градиентного спуска с lr = 0.05.
w, b, lr, hist = 0.0, 0.0, 0.05, []

for i in range(200):
    dw, db = grads(w, b)
    w -= lr * dw
    b -= lr * db
    hist.append(loss(w, b))

print('Найденные w и b:', w, b)

# График функции потерь — должна падать и выравниваться.
plt.plot(hist)
plt.show()

# Прямая поверх точек данных.
plt.scatter(X, y)
plt.plot(X, w * X + b, 'r')
plt.show()


# ============================================================================
# Шаг 7. Сверка с sklearn
# ============================================================================
# Внутри sklearn LinearRegression делает то же самое — ищет w и b,
# минимизируя MSE. coef_ и intercept_ должны совпадать с найденными вручную.
from sklearn.linear_model import LinearRegression

sklearn_lr = LinearRegression()
sklearn_lr.fit(X.reshape(-1, 1), y)

print('sklearn coef_:', sklearn_lr.coef_[0])
print('sklearn intercept_:', sklearn_lr.intercept_)