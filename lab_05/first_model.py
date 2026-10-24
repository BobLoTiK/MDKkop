import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# UTF-8 для консоли
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
CLEAN_CSV = HERE / 'data' / 'clean.csv'

# Шаг 1–3. k ближайших соседей на ирисах

# Загружаем встроенный датасет ирисов: 4 признака, 3 класса.
X, y = load_iris(return_X_y=True)

# Делим на обучение и тест: 75% / 25%, фиксируем random_state для воспроизводимости.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=0)

# Обучаем kNN с k=3.
knn = KNeighborsClassifier(3).fit(X_tr, y_tr)

# Предсказываем на тесте.
pred = knn.predict(X_te)

# Выводим первые 5 предсказаний рядом с правдой и accuracy.
print(pred[:5], y_te[:5], 'accuracy:', accuracy_score(y_te, pred))

# Шаг 4. Titanic: предсказать выживание

# Читаем очищенный CSV из задачи 4.
df = pd.read_csv(CLEAN_CSV)

# Признаки по заданию: pclass, age, fare, family, sex_num; цель — survived.
feats = ['pclass', 'age', 'fare', 'family', 'sex_num']

# Делим на обучение и тест 75/25.
X_tr, X_te, y_tr, y_te = train_test_split(df[feats], df['survived'], test_size=.25, random_state=0)

# Обучаем логистическую регрессию.
lr = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)

# accuracy на тесте и базовая модель «все не выжили».
print('accuracy:', lr.score(X_te, y_te), 'baseline:', (y_te == 0).mean())

# Шаг 5. Матрица ошибок

# Где модель ошибается: precision, recall, f1.
print(confusion_matrix(y_te, lr.predict(X_te)))
print(classification_report(y_te, lr.predict(X_te)))

# Шаг 6. Новый пассажир

# Вымышленный пассажир: класс, возраст, цена билета, размер семьи, пол.
new = pd.DataFrame([[1, 25, 80, 0, 1]], columns=feats)

# predict — класс (выжил/не выжил), predict_proba — вероятность каждого класса.
print(lr.predict(new), lr.predict_proba(new))