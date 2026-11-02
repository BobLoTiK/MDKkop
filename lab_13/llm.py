import sys
import json

import requests

# UTF-8 для консоли
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

MODEL = 'qwen2.5:0.5b'          # одна правка — и модель 1.5b
URL = 'http://localhost:11434/api/generate'


def ask(prompt, system='', temperature=0.7):
    r = requests.post(URL, json={
        'model': MODEL, 'prompt': prompt, 'system': system, 'stream': False,
        'options': {'temperature': temperature}})
    return r.json()['response']


def ask_json(prompt, system=''):
    r = requests.post(URL, json={
        'model': MODEL, 'prompt': prompt, 'system': system, 'stream': False,
        'format': 'json'})
    return json.loads(r.json()['response'])


def main():
    # Ollama должна быть запущена, модель скачана: ollama pull qwen2.5:0.5b
    try:
        requests.get('http://localhost:11434/api/tags', timeout=3)
    except requests.exceptions.RequestException:
        print('Нет соединения с Ollama на localhost:11434 — запусти ollama serve')
        return

    # Шаг 1. Размер и квантизация — 10 мин
    # ollama list: 0.5b ≈ 400 МБ; 0.5 млрд × 2 байта = 1 ГБ в fp16, значит веса сжаты.
    # ollama show qwen2.5:0.5b покажет квантизацию и размер контекста. (команды — в терминале)

    # Шаг 2. Функция ask() — 15 мин
    print('— шаг 2 —')
    print(ask('Объясни, что такое нейросеть, в двух предложениях.'))

    # Шаг 3. Температура — 15 мин
    # При 0 ответы одинаковые, при 1.5 — разные и иногда странные.
    print('\n— шаг 3: температура 0 —')
    q = 'Назови три причины учить numpy.'
    cold = [ask(q, temperature=0) for _ in range(3)]
    print(cold[0])
    print('одинаковых ответов:', len({c.strip() for c in cold}), 'из 3')
    print('— шаг 3: температура 1.5 —')
    hot = [ask(q, temperature=1.5) for _ in range(3)]
    for h in hot:
        print(' •', h.strip().replace('\n', ' ')[:120])

    # Шаг 4. Системный промпт — 15 мин
    print('\n— шаг 4: без системного промпта —')
    print(ask('Что делать, если не сдал лабораторную?'))
    print('— шаг 4: с системным промптом —')
    print(ask('Что делать, если не сдал лабораторную?',
              system='Ты ассистент колледжа. Отвечай кратко, вежливо, только по-русски.'))
    print('— шаг 4: «Отвечай только ДА или НЕТ» —')
    print(ask('Нужно ли делать домашку?', system='Отвечай только ДА или НЕТ'))

    # Шаг 5. Few-shot классификация — 15 мин
    print('\n— шаг 5: few-shot —')
    p = '''Определи категорию обращения одним словом.
Фраза: Где расписание на завтра? Категория: расписание
Фраза: Столовая сегодня работает? Категория: столовая
Фраза: Мне нужна справка для военкомата. Категория: документы
Фраза: Во сколько закрывается буфет? Категория:'''
    print(ask(p, temperature=0))

    # Шаг 6. JSON — 15 мин
    print('\n— шаг 6: JSON —')
    r = requests.post(URL, json={
        'model': MODEL, 'stream': False, 'format': 'json',
        'prompt': 'Верни JSON с полями intent (одно слово) и answer (одно предложение) для запроса: Где столовая?'})
    print(json.loads(r.json()['response']))

    # Шаг 7. Ограничения — 5 мин
    # Спросим то, чего модель знать не может: сегодняшнюю дату и своё расписание.
    print('\n— шаг 7: чего модель не знает —')
    print(ask('Какое сегодня число?', temperature=0))
    print(ask('Какое расписание занятий у группы МДК на завтра?', temperature=0))


if __name__ == '__main__':
    main()
