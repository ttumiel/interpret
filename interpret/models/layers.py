import torch
from torch import nn


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x)
    def __repr__(self):
        return f"Lambda({self.fn.__name__})"
