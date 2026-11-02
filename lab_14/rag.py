import sys
from pathlib import Path

import numpy as np

# UTF-8 для консоли
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent

# Шаг 1. Корпус — 10 мин
# docs.txt — 2–3 страницы своего текста (правила, описание курса, конкурсное задание).
text = open(HERE / 'docs.txt', encoding='utf-8').read()
chunks = [text[i:i+400] for i in range(0, len(text), 350)]
print(len(chunks)); print(chunks[0])

# Шаг 2. Эмбеддинги — 15 мин
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer('cointegrated/rubert-tiny2')
V = emb.encode(chunks, normalize_embeddings=True)
print(V.shape)

# Шаг 3. Близость на глаз — 10 мин
# Две фразы про одно и та же — близкие (0.7+), третья про другое — ниже 0.4.
Q = emb.encode(['Сколько часов длится конкурс?', 'Какова длительность конкурса?', 'Во сколько открывается столовая?'],
               normalize_embeddings=True)
print(np.round(Q @ Q.T, 2))

# Шаг 4. Поиск — 15 мин
def search(q, k=3):
    qv = emb.encode([q], normalize_embeddings=True)[0]
    scores = V @ qv
    idx = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[i]) for i in idx]

for s, c in search('Сколько часов длится конкурс?'):
    print(round(s, 3), c[:100])

# Шаг 5. Генерация по контексту — 20 мин
# ask() берём из задачи 13 — llm.py лежит в соседней папке.
sys.path.insert(0, str(HERE.parent / 'lab_13'))
from llm import ask


def rag(q, k=2):
    # Для маленькой модели 0.5b держим k=2, иначе контекст её путает.
    ctx = '\n---\n'.join(c for _, c in search(q, k))
    p = f'Ответь на вопрос только по контексту. Если ответа в контексте нет, скажи: не знаю.\n\nКонтекст:\n{ctx}\n\nВопрос: {q}\nОтвет:'
    return ask(p, temperature=0)


print()
print('— вопрос по корпусу —')
print(rag('Сколько часов длится конкурс?'))
print('— вопрос вне корпуса —')
print(rag('Какая столица Австралии?'))

# Шаг 6. Оценка — 15 мин
# 10 вопросов, у каждого известный кусок-ответ (проверяем по ключевому слову).
qa = [
    ('Сколько часов длится конкурс?', 'шесть'),
    ('Во сколько открывается столовая?', '9:00'),
    ('Когда дедлайн лабораторных?', '25 декабря'),
    ('Из скольких задач состоит курс?', 'пятнадцати'),
    ('Какой порог accuracy для MNIST?', '97'),
    ('В какой аудитории консультации?', '214'),
    ('Сколько человек в команде на конкурсе?', 'трёх'),
    ('Какая модель LLM работает локально?', 'qwen2.5'),
    ('Что нужно сдать в финале?', 'аватар'),
    ('Сколько памяти нужно ноутбуку?', '8 ГБ'),
]
hits = 0
print()
print(f'{"#":<3} {"top1":>5} {"в топ-3":>7}  вопрос')
for i, (q, gold) in enumerate(qa, 1):
    top = search(q, 3)
    hit = any(gold.lower() in c.lower() for _, c in top)
    hits += hit
    print(f'{i:<3} {top[0][0]:>5.2f} {("да" if hit else "нет"):>7}  {q}')
print('recall@3:', round(hits / len(qa), 2))

print()
print('ответы LLM (0/1 руками):')
for q, _ in qa:
    print(' •', q, '→', rag(q).strip().replace('\n', ' ')[:140])
