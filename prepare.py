import sys
from pathlib import Path

import pandas as pd

# Пути и кодировка: скрипт запускается из любой папки и печатает русский текст
# в любой консоли Windows (cmd, PowerShell, терминал VS Code).
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent   # папка, в которой лежит сам скрипт
CSV = HERE / 'data' / 'titanic.csv'      # данные лежат в подпапке data
if not CSV.exists():                     # запасной вариант: csv рядом со скриптом
    CSV = HERE / 'titanic.csv'
OUT_CSV = CSV.parent / 'clean.csv'       # готовую таблицу кладём рядом с исходной

df = pd.read_csv(CSV)
print(df.shape); print(df.head()); df.info(); print(df.describe())
print("______")

kids = df[df['age'] < 12]
print(len(kids), kids['survived'].mean())
print("______")

df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = df.drop(columns=['deck'])
print(df.isna().sum())
print("______")

print(df.groupby('sex')['survived'].mean())
print(df.groupby('pclass')['survived'].mean())
print(df.groupby(['pclass', 'sex'])['age'].mean().round(1))
print("______")

import matplotlib.pyplot as plt, seaborn as sns
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
ax[0].hist(df['age'], bins=30); ax[0].set_title('Возраст')
sns.boxplot(x='pclass', y='fare', data=df, ax=ax[1]); ax[1].set_title('Цена по классу')
sns.barplot(x='sex', y='survived', data=df, ax=ax[2]); ax[2].set_title('Выживаемость')
fig.tight_layout(); fig.savefig(HERE / 'titanic.png', dpi=120)
plt.show()
print("______")

df['family'] = df['sibsp'] + df['parch']
df['sex_num'] = df['sex'].map({'female': 1, 'male': 0})
df.to_csv(OUT_CSV, index=False)

print("______")

clean = pd.read_csv(OUT_CSV)
print(clean.isna().sum())

print("______")

plt.hist(df[df['survived'] == 1]['fare'], bins=30, alpha=0.5, label='Выжившие')
plt.hist(df[df['survived'] == 0]['fare'], bins=30, alpha=0.5, label='Невыжившие')
plt.xlabel('Цена билета') 
plt.ylabel('Количество пассажиров')
plt.title('Цена билета для выживших и невыживших')
plt.legend()
plt.show()
