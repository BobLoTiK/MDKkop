import sys
from pathlib import Path

# Переключаем консоль на UTF-8, чтобы русский текст не отображался кракозябрами
# в разных терминалах (cmd, PowerShell, VS Code).
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

# Находим файл с данными относительно расположения этого скрипта,
# чтобы запуск работал из любой папки.
HERE = Path(__file__).resolve().parent
CSV = HERE / 'data' / 'titanic.csv'
if not CSV.exists():
    CSV = HERE / 'titanic.csv'

# Показ разных встроенных типов Python: число, строка, список, словарь.
x = 5
name = 'Аня'
nums = [3, 1, 4, 1, 5]
person = {'name': 'Аня', 'age': 19}

print(type(x), type(name), type(nums), type(person))
print(nums[0], nums[-1], len(nums), sum(nums) / len(nums))
print(person['name'])


def mean(lst):
    # Среднее арифметическое, написанное вручную без библиотек.
    total = 0
    for v in lst:
        total += v
    return total / len(lst)


def minmax(lst):
    # Минимум и максимум за один проход по списку.
    lo, hi = lst[0], lst[0]
    for v in lst:
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    return lo, hi


print(mean([3, 1, 4]), minmax([3, 1, 4]))

# Читаем CSV без pandas — просто открываем файл и парсим строки.
with open(CSV, encoding='utf-8') as f:
    header = f.readline().strip().split(',')   # первая строка — имена колонок
    rows = [line.strip().split(',') for line in f]   # остальные — данные

print(header)
print(len(rows), 'строк')
print(rows[0])

# Разбираем таблицу по колонкам: каждое имя столбца — ключ, значение — список данных.
cols = {h: [] for h in header}
for r in rows:
    for h, v in zip(header, r):
        if v == '':
            continue   # пропускаем пустые ячейки
        try:
            cols[h].append(float(v))   # пробуем сохранить как число
        except ValueError:
            cols[h].append(v)   # если не число — оставляем текстом

print(cols['age'][:10])

# Выводим статистику по каждой колонке: для чисел — среднее, минимум, максимум;
# для текста — количество уникальных значений.
for h, vals in cols.items():
    if not vals:
        continue
    if isinstance(vals[0], float):
        lo, hi = minmax(vals)
        print(f'{h:12s} n={len(vals):4d} mean={mean(vals):8.2f} min={lo:6.1f} max={hi:6.1f}')
    else:
        print(f'{h:12s} n={len(vals):4d} unique={len(set(vals))}')


def median(lst):
    # Медиана: средний элемент после сортировки (или平均 двух центральных при чётном).
    s = sorted(lst)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    else:
        return (s[n // 2 - 1] + s[n // 2]) / 2


print('Медиана возраста:', median(cols['age']))