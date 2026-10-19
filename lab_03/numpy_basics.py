import sys
from pathlib import Path

import numpy as np

# Переключаем консоль на UTF-8 для корректного вывода русского текста.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

# Находим файл с данными относительно расположения этого скрипта.
HERE = Path(__file__).resolve().parent
CSV = HERE / 'data' / 'titanic.csv'
if not CSV.exists():
    CSV = HERE / 'titanic.csv'

# Одномерный массив: арифметика применяется к каждому элементу отдельно.
a = np.array([3, 1, 4, 1, 5])
print(a * 2, a + 10, a ** 2)
print(a.mean(), a.min(), a.max(), a.sum())
print("______")

# Двумерный массив (матрица): форма, срезы по строкам и столбцам, транспонирование.
M = np.arange(12).reshape(3, 4)
print(M, M.shape)
print('строка 0:', M[0])
print('столбец 1:', M[:, 1])
print('элемент:', M[2, 3])
print(M.T.shape)
print("______")

# Среднее по осям: axis=0 — по столбцам, axis=1 — по строкам.
print(M.mean(axis=0))
print(M.mean(axis=1))
print("______")

# Матричное умножение и Broadcasting: прибавление вектора к каждой строке матрицы.
A = np.array([[1, 2, 3], [4, 5, 6]])
w = np.array([[1], [0], [-1]])
print(A @ w)
print(A + np.array([10, 20, 30]))
print("______")

# Пример линейной модели y = X @ w + b — основа работы нейросетей.
rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5))   # 100 объектов, 5 признаков
w = np.array([0.5, -1.0, 2.0, 0.0, 1.5])   # веса
b = 0.3   # смещение
y = X @ w + b
print(X.shape, w.shape, y.shape, y[:5])
print("______")

# Читаем Titanic вручную (без pandas) и считаем статистику по возрасту.
with open(CSV, encoding='utf-8') as f:
    header = f.readline().strip().split(',')
    rows = [line.strip().split(',') for line in f]

cols = {h: [] for h in header}
for r in rows:
    for h, v in zip(header, r):
        if v == '':
            continue
        try:
            cols[h].append(float(v))
        except ValueError:
            cols[h].append(v)

age = np.array(cols['age'])
print(age.mean(), age.min(), age.max())