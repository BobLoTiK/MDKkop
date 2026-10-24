import sys
from pathlib import Path

import pandas as pd

# Переключаем консоль на UTF-8 для корректного вывода русского текста.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

# Находим файл с данными относительно расположения этого скрипта.
HERE = Path(__file__).resolve().parent
CSV = HERE / 'data' / 'titanic.csv'
if not CSV.exists():
    CSV = HERE / 'titanic.csv'
OUT_CSV = CSV.parent / 'clean.csv'

# Загружаем таблицу и делаем первичный осмотр: размер, первые строки, информация о типах и базовая статистика.
df = pd.read_csv(CSV)
print(df.shape)
print(df.head())
df.info()
print(df.describe())
print("______")

# Пассажиры младше 12 лет и доля выживших среди них.
kids = df[df['age'] < 12]
print(len(kids), kids['survived'].mean())
print("______")

# Очистка данных: заполняем пропуски возраста медианой, порта посадки — наиболее частым значением, удаляем колонку с demasiado большим количеством пропусков.
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = df.drop(columns=['deck'])
print(df.isna().sum())
print("______")

# Группировка: как зависит выживаемость от пола и класса, средний возраст по сочетанию класс + пол.
print(df.groupby('sex')['survived'].mean())
print(df.groupby('pclass')['survived'].mean())
print(df.groupby(['pclass', 'sex'])['age'].mean().round(1))
print("______")

# Визуализация: три графика рядом — возраст, выживаемость по классам, доля выживших по полу.
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
ax[0].hist(df['age'], bins=30)
ax[0].set_title('Возраст')
sns.barplot(x='pclass', y='survived', data=df, ax=ax[1])
ax[1].set_title('Выживаемость по классам')
sns.barplot(x='sex', y='survived', data=df, ax=ax[2])
ax[2].set_title('Выживаемость')

fig.tight_layout()
fig.savefig(HERE / 'titanic.png', dpi=120)
plt.show()
print("______")

# Инженерия признаков: размер семьи и пол в числовом виде, сохраняем в clean.csv.
df['family'] = df['sibsp'] + df['parch']
df['sex_num'] = df['sex'].map({'female': 1, 'male': 0})
df.to_csv(OUT_CSV, index=False)

print("______")

# Проверка, что в сохранённом файле не осталось пропусков.
clean = pd.read_csv(OUT_CSV)
print(clean.isna().sum())
print("______")

# Гистограмма цены билета для выживших и невыживших — сравнение распределений.
plt.hist(df[df['survived'] == 1]['fare'], bins=30, alpha=0.5, label='Выжившие')
plt.hist(df[df['survived'] == 0]['fare'], bins=30, alpha=0.5, label='Невыжившие')
plt.xlabel('Цена билета')
plt.ylabel('Количество пассажиров')
plt.title('Цена билета для выживших и невыживших')
plt.legend()
plt.show()
