import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
torch.set_num_threads(4)

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train = datasets.MNIST(str(DATA), train=True, download=True, transform=tf)
test = datasets.MNIST(str(DATA), train=False, download=True, transform=tf)
train_dl = DataLoader(train, batch_size=128, shuffle=True)
test_dl = DataLoader(test, batch_size=512)

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

for ep in range(3):
    t0 = time.time(); model.train()
    for x, y in train_dl:
        opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in test_dl: correct += (model(x).argmax(1) == y).sum().item()
    sched.step()
    print(f'epoch {ep} loss {loss.item():.3f} acc {correct / len(test):.4f} time {time.time() - t0:.0f}s')

torch.save(model.state_dict(), HERE / 'mnist.pt')
print('сохранено:', (HERE / 'mnist.pt').name, (HERE / 'mnist.pt').stat().st_size, 'байт')
