"""Быстрый запуск лабораторных работ из корня репозитория.

    python run.py           — список лаб
    python run.py 7         — запустить лабу 7 (её главный скрипт)
    python run.py lab7      — то же самое
    python run.py 15 app    — конкретный файл внутри лабы (train / app / predict)
    python run.py 12 ../lab_11/photo.jpg — аргументы уходят самому скрипту

Ничего не копируется и не переименовывается: путь к скрипту ищется по папке
lab_NN, запуск идёт из этой папки, поэтому относительные пути внутри лабы
работают так же, как при обычном `python neuron.py`.
"""

import subprocess
import sys
from pathlib import Path

# В cmd.exe кодировка консоли 866, а Python по умолчанию пишет cp1251 —
# русский текст превращается в кракозябры. Пишем в UTF-8, как и сами лабы.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent

# Главный скрипт каждой лабы. Остальные файлы лабы доступны по имени:
# python run.py 15 train — запустит lab_15/project15/train.py.
PRIMARY = {
    1: None,               # лаба 1 — лекция, скрипта нет
    2: 'stats',
    3: 'numpy_basics',
    4: 'prepare',
    5: 'first_model',
    6: 'gd',
    7: 'neuron',
    8: 'torch_neuron',
    9: 'overfit',
    10: 'train_mnist',
    11: 'pretrained',
    12: 'ants_bees',
    13: 'llm',
    14: 'rag',
    15: 'app',             # интерфейс Gradio; python run.py 15 train — обучение
}

TITLES = {
    2: 'статистика по CSV руками',
    3: 'numpy: массивы и формы',
    4: 'pandas: пропуски, группировки, графики',
    5: 'первая модель (kNN, логистическая регрессия)',
    6: 'градиентный спуск руками',
    7: 'один нейрон на numpy',
    8: 'тот же нейрон на PyTorch',
    9: 'переобучение и три лечения',
    10: 'MNIST на CPU',
    11: 'готовые модели (текст, перевод, фото)',
    12: 'признаки MobileNet + голова',
    13: 'Ollama: ask(), температура, JSON',
    14: 'мини-RAG по своим документам',
    15: 'итог: модель + интерфейс + README',
}


def lab_dir(number):
    return ROOT / f'lab_{number:02d}'


def scripts_in(number):
    d = lab_dir(number)
    if not d.is_dir():
        return {}
    return {p.stem: p for p in sorted(d.rglob('*.py')) if '__pycache__' not in p.parts}


def parse_number(arg):
    text = arg.lower().replace('lab_', '').replace('lab', '').replace('-', '').strip()
    if not text.isdigit():
        raise SystemExit(f'не понял номер лабы: {arg!r}. Пример: python run.py 7')
    return int(text)


def show_list():
    print('лабы в', ROOT)
    print()
    for n in sorted(PRIMARY):
        scripts = scripts_in(n)
        if not lab_dir(n).is_dir():
            print(f'  {n:>2}. (нет папки lab_{n:02d})')
            continue
        if not scripts:
            print(f'  {n:>2}. лекция, скрипта нет')
            continue
        main = PRIMARY[n]
        mark = scripts[main].relative_to(ROOT) if main in scripts else next(iter(scripts.values())).relative_to(ROOT)
        extra = sorted(s for s in scripts if s != main)
        tail = f'   ещё: {", ".join(extra)}' if extra else ''
        print(f'  {n:>2}. {mark}{tail}')
        print(f'      {TITLES.get(n, "")}')
    print()
    print('запуск: python run.py <номер> [имя_файла] [аргументы]')


def main(argv):
    if not argv or argv[0].lower() in ('list', 'ls', '-l', '--list'):
        show_list()
        return 0

    number = parse_number(argv[0])
    rest = argv[1:]

    if number == 1:
        print('Лаба 01 — лекция «Дистилляция и квантизация», скрипта в задании нет.')
        return 0

    scripts = scripts_in(number)

    if not scripts:
        print(f'В lab_{number:02d} нет python-скриптов.')
        return 1

    name = None
    if rest and not rest[0].startswith('-') and rest[0] in scripts:
        name = rest[0]
        rest = rest[1:]

    script = scripts[name] if name else scripts.get(PRIMARY.get(number))
    if script is None:
        print(f'В lab_{number:02d} несколько скриптов: {", ".join(sorted(scripts))}.')
        print(f'Укажи имя: python run.py {number} <имя>')
        return 1

    print(f'→ {script.relative_to(ROOT)}', flush=True)
    return subprocess.call([sys.executable, str(script), *rest], cwd=script.parent)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
