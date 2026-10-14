import os
import sys

import requests

# Кодировка: русский ответ модели и сообщения об ошибках печатаются в любой консоли.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

OLLAMA_URL = 'http://localhost:11434/api/generate'
# Модель по умолчанию; можно переопределить без правки файла:
#   PowerShell:  $env:OLLAMA_MODEL='qwen2.5:1.5b-instruct'; python hello_llm.py
MODEL = os.environ.get('OLLAMA_MODEL', 'qwen2.5:1.5b')

try:
    r = requests.post(OLLAMA_URL, json={
        'model': MODEL,
        'prompt': 'Что такое нейросеть? Ответь в двух предложениях по-русски.',
        'stream': False}, timeout=180)
    r.raise_for_status()
except requests.exceptions.ConnectionError:
    print('Нет соединения с Ollama на localhost:11434.')
    print('Сначала запусти Ollama (команда: ollama serve), потом этот скрипт.')
    raise SystemExit(1)
except requests.exceptions.HTTPError:
    print(f'Ollama ответила ошибкой: {r.text[:300]}')
    print(f'Скорее всего модель "{MODEL}" не скачана.')
    print('Что уже есть локально:  ollama list')
    print(f'Скачать нужную:         ollama pull {MODEL}')
    print('Или подставить другую из списка:')
    print("  PowerShell: $env:OLLAMA_MODEL='qwen2.5:1.5b-instruct'; python hello_llm.py")
    raise SystemExit(1)

print(r.json()['response'])