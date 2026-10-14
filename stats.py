import sys
from pathlib import Path

# Пути и кодировка: скрипт запускается из любой папки и печатает русский текст
# в любой консоли Windows (cmd, PowerShell, терминал VS Code).
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent   # папка, в которой лежит сам скрипт
CSV = HERE / 'data' / 'titanic.csv'      # данные лежат в подпапке data
if not CSV.exists():                     # запасной вариант: csv рядом со скриптом
    CSV = HERE / 'titanic.csv'


x = 5
name = 'Аня'
nums = [3, 1, 4, 1, 5]
person = {'name': 'Аня', 'age': 19}
print(type(x), type(name), type(nums), type(person))
print(nums[0], nums[-1], len(nums), sum(nums) / len(nums))
print(person['name'])

def mean(lst):
    total = 0
    for v in lst:
        total += v
    return total / len(lst)
 
def minmax(lst):
    lo, hi = lst[0], lst[0]
    for v in lst:
        if v < lo: lo = v
        if v > hi: hi = v
    return lo, hi
 
print(mean([3, 1, 4]), minmax([3, 1, 4]))

with open(CSV, encoding='utf-8') as f:
    header = f.readline().strip().split(',')
    rows = [line.strip().split(',') for line in f]
print(header)
print(len(rows), 'строк')
print(rows[0])

cols = {h: [] for h in header}
for r in rows:
    for h, v in zip(header, r):
        if v == '':
            continue
        try:
            cols[h].append(float(v))
        except ValueError:
            cols[h].append(v)
print(cols['age'][:10])

for h, vals in cols.items():
    if not vals:
        continue
    if isinstance(vals[0], float):
        lo, hi = minmax(vals)
        print(f'{h:12s} n={len(vals):4d} mean={mean(vals):8.2f} min={lo:6.1f} max={hi:6.1f}')
    else:
        print(f'{h:12s} n={len(vals):4d} unique={len(set(vals))}')
def median(lst):
    s = sorted(lst)
    n = len(s)

    if n % 2 == 1:
        return s[n // 2]
    else:
        return (s[n // 2 - 1] + s[n // 2]) / 2
print('Медиана возраста:', median(cols['age']))