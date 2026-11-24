"""Architectures shared by trainer and inference endpoint."""
import torch
from torch import nn


def make_model(task: str, input_dim: int = 0, classes: int = 3):
    if task == 'synthetic_images':
        return nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 8 * 8, 48), nn.ReLU(), nn.Linear(48, 3),
        )
    if task == 'csv':
        return nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32),
                             nn.ReLU(), nn.Linear(32, classes))
    raise ValueError('Unknown task: ' + task)


def make_images(n: int, seed: int):
    """3 genuinely distinct 32x32 image classes; no download or external network."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((n, 1, 32, 32), generator=gen) * 0.12
    y = torch.arange(n, dtype=torch.long) % 3
    for i in range(n):
        c = int(y[i])
        if c == 0:
            x[i, :, :, 8:13] += 1.3
            x[i, :, :, 20:25] += 0.9
        elif c == 1:
            x[i, :, 8:13, :] += 1.3
            x[i, :, 20:25, :] += 0.9
        else:
            ids = torch.arange(32)
            for width in [-2, -1, 0, 1, 2]:
                x[i, 0, ids, (ids + width) % 32] += 1.2
    order = torch.randperm(n, generator=gen)
    return x[order], y[order]
