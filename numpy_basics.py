import sys
from pathlib import Path

import numpy as np

# Пути и кодировка: скрипт запускается из любой папки и печатает русский текст
# в любой консоли Windows (cmd, PowerShell, терминал VS Code).
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent   # папка, в которой лежит сам скрипт
CSV = HERE / 'data' / 'titanic.csv'      # данные лежат в подпапке data
if not CSV.exists():                     # запасной вариант: csv рядом со скриптом
    CSV = HERE / 'titanic.csv'

a = np.array([3, 1, 4, 1, 5])

print(a * 2, a + 10, a ** 2)
print(a.mean(), a.min(), a.max(), a.sum())
print("______")
M = np.arange(12).reshape(3, 4)

print(M, M.shape)
print('строка 0:', M[0])
print('столбец 1:', M[:, 1])
print('элемент:', M[2, 3])

print(M.T.shape)
print("______")

print(M.mean(axis=0))
print(M.mean(axis=1))
print("______")
A = np.array([[1, 2, 3], [4, 5, 6]])

w = np.array([[1], [0], [-1]])

print(A @ w)
print(A + np.array([10, 20, 30]))
print("______")
rng = np.random.default_rng(0)

X = rng.normal(size=(100, 5))
w = np.array([0.5, -1.0, 2.0, 0.0, 1.5])
b = 0.3

y = X @ w + b

print(X.shape, w.shape, y.shape, y[:5])
print("______")
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