import sys
import copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# UTF-8 для консоли
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
torch.manual_seed(0)

# Шаг 1. Полиномы 1, 4, 15 — 20 мин
rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 1, 30)); y = np.sin(2 * np.pi * X) + rng.normal(0, .3, 30)
Xv = np.sort(rng.uniform(0, 1, 30)); yv = np.sin(2 * np.pi * Xv) + rng.normal(0, .3, 30)
print('степень  MSE train  MSE val')
for d in [1, 4, 15]:
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(X[:, None], y)
    print(f'{d:>7}  {np.mean((m.predict(X[:, None]) - y)**2):.4f}  {np.mean((m.predict(Xv[:, None]) - yv)**2):.4f}')

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
fig.suptitle('подгонка полинома разной степени: 1, 4, 15')
for ax, d in zip(axes, [1, 4, 15]):
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(X[:, None], y)
    xs = np.linspace(0, 1, 200)
    ax.scatter(X, y, s=15, label='train'); ax.scatter(Xv, yv, s=15, marker='x', label='val')
    ax.plot(xs, m.predict(xs[:, None]), 'r', label=f'степень {d}')
    ax.set_ylim(-2, 2); ax.legend(fontsize=8); ax.set_title(f'полином {d}')
plt.tight_layout(); plt.savefig(HERE / 'poly.png', bbox_inches='tight'); plt.show()

# Шаг 2. Сеть переобучается на moons — 20 мин
# 100 точек noise 0.3, split 50/50. make_moons отдаёт точки по классам подряд,
# поэтому делим после перемешивания — иначе val был бы целиком из одного класса.
Xm, ym = make_moons(100, noise=0.3, random_state=0)
idx = np.random.default_rng(0).permutation(len(Xm))
Xtr0 = torch.tensor(Xm[idx[:50]], dtype=torch.float32); ytr0 = torch.tensor(ym[idx[:50]], dtype=torch.float32)
Xva0 = torch.tensor(Xm[idx[50:]], dtype=torch.float32); yva0 = torch.tensor(ym[idx[50:]], dtype=torch.float32)
Xtr, ytr, Xva, yva = Xtr0, ytr0, Xva0, yva0
loss_fn = nn.BCEWithLogitsLoss()


def fit(model, opt, steps=2000, patience=None):
    """Цикл из шага 2. patience=None — train/val каждые 50 шагов,
    patience=10 — ранняя остановка из шага 3 (10 проверок без улучшения)."""
    tr, va, best, best_va, bad = [], [], None, float('inf'), 0
    for i in range(steps):
        model.train(); opt.zero_grad(); l = loss_fn(model(Xtr).squeeze(1), ytr); l.backward(); opt.step()
        if i % 50 == 0:
            model.eval()
            with torch.no_grad():
                tr.append(l.item()); va.append(loss_fn(model(Xva).squeeze(1), yva).item())
            if va[-1] < best_va:
                best_va, best, bad = va[-1], copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if patience and bad >= patience:
                    break
    return tr, va, best, best_va


def big_net():
    return nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,1))


def val_acc(model):
    model.eval()
    with torch.no_grad():
        return ((model(Xva).squeeze(1) > 0).float() == yva).float().mean().item()


big = big_net()
tr, va, best_state, best_val = fit(big, torch.optim.Adam(big.parameters(), 1e-2))
acc_big = val_acc(big)
print()
print(f'без регуляризации: val loss {va[0]:.3f} → минимум {best_val:.3f} → {va[-1]:.3f} на шаге {50 * len(va)} (val acc {acc_big:.3f})')
plt.figure(figsize=(6, 4))
plt.plot(tr, label='train'); plt.plot(va, label='val'); plt.legend(); plt.title('переобучение: val растёт, train падает')
plt.savefig(HERE / 'overfit.png', bbox_inches='tight'); plt.show()

# Шаг 3. Лечение 1: ранняя остановка — 10 мин
es = big_net()
tr_es, va_es, best_es, best_es_val = fit(es, torch.optim.Adam(es.parameters(), 1e-2), patience=10)
es.load_state_dict(best_es)          # лучшие веса: state_dict из copy.deepcopy
acc_es = val_acc(es)
print(f'ранняя остановка: останов на шаге {50 * len(va_es)} из 2000, val loss {best_es_val:.4f}, val acc {acc_es:.3f}')

# Шаг 4. Лечение 2: weight decay — 10 мин
wd = big_net()
tr_wd, va_wd, _, best_wd = fit(wd, torch.optim.Adam(wd.parameters(), 1e-2, weight_decay=1e-2))
acc_wd = val_acc(wd)
print(f'weight decay: val loss {best_wd:.4f}, val acc {acc_wd:.3f}')

# Шаг 5. Лечение 3: dropout — 15 мин
reg = nn.Sequential(nn.Linear(2,128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128,128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128,1))
tr2, va2, _, best_reg = fit(reg, torch.optim.Adam(reg.parameters(), 1e-2, weight_decay=1e-2))
acc_reg = val_acc(reg)
print(f'dropout: val loss {best_reg:.4f}, val acc {acc_reg:.3f}')

# Шаг 6. Больше данных — 10 мин
# 2000 точек: 1000 train / 1000 val, та же сеть. Здесь свой val, поэтому
# val loss сравнивать с предыдущими строками напрямую нельзя.
Xm2, ym2 = make_moons(2000, noise=0.3, random_state=0)
idx2 = np.random.default_rng(0).permutation(len(Xm2))
Xtr = torch.tensor(Xm2[idx2[:1000]], dtype=torch.float32); ytr = torch.tensor(ym2[idx2[:1000]], dtype=torch.float32)
Xva = torch.tensor(Xm2[idx2[1000:]], dtype=torch.float32); yva = torch.tensor(ym2[idx2[1000:]], dtype=torch.float32)
more = big_net()
tr_more, va_more, _, best_more = fit(more, torch.optim.Adam(more.parameters(), 1e-2))
acc_more = val_acc(more)
print(f'больше данных: val loss {best_more:.4f}, val acc {acc_more:.3f} (val на 1000 точках)')

fig, axes = plt.subplots(2, 2, figsize=(11, 6))
fig.suptitle('три лечения переобучения и проверка «больше данных»')
for ax, (a, b, t) in zip(axes.flat, [
    (tr_es, va_es, 'ранняя остановка'),
    (tr_wd, va_wd, 'weight decay'),
    (tr2, va2, 'dropout'),
    (tr_more, va_more, 'больше данных (2000 точек)'),
]):
    ax.plot(a, label='train'); ax.plot(b, label='val'); ax.set_title(t); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(HERE / 'cures.png', bbox_inches='tight'); plt.show()

# Шаг 7. Таблица — 5 мин
print()
print('вариант                 лучший val loss  val acc')
for name, loss_v, acc_v in [
    ('без регуляризации', best_val, acc_big),
    ('ранняя остановка', best_es_val, acc_es),
    ('weight decay', best_wd, acc_wd),
    ('dropout', best_reg, acc_reg),
    ('больше данных', best_more, acc_more),
]:
    print(f'{name:<23} {loss_v:>14.4f} {acc_v:>8.3f}')

# Домашнее повторение: ReduceLROnPlateau по val loss
# Возвращаемся на исходные 50/50 точек, чтобы сравнивать с шагом 2.
Xtr, ytr, Xva, yva = Xtr0, ytr0, Xva0, yva0
rl = big_net()
opt = torch.optim.Adam(rl.parameters(), 1e-2)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2)
tr_rl, va_rl, best_rl = [], [], float('inf')
for i in range(2000):
    rl.train(); opt.zero_grad(); l = loss_fn(rl(Xtr).squeeze(1), ytr); l.backward(); opt.step()
    if i % 50 == 0:
        rl.eval()
        with torch.no_grad():
            tr_rl.append(l.item()); v = loss_fn(rl(Xva).squeeze(1), yva).item(); va_rl.append(v)
        sched.step(v); best_rl = min(best_rl, v)
print()
print(f'ReduceLROnPlateau: лучший val loss {best_rl:.4f} (было {best_val:.4f}), итоговый lr {opt.param_groups[0]["lr"]:.1e}')
plt.figure(figsize=(6, 4))
plt.plot(tr_rl, label='train'); plt.plot(va_rl, label='val'); plt.title('ReduceLROnPlateau (домашка)'); plt.legend()
plt.savefig(HERE / 'plateau.png', bbox_inches='tight'); plt.show()
