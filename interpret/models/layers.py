import torch
from torch import nn

__all__ = ['Lambda', 'GeneralizedReLU']

class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x)
    def __repr__(self):
        return f"Lambda({self.fn.__name__})"


class GeneralizedReLU(nn.Module):
    def __init__(self, leak=None, bias=0, max_value=None, inplace=False):
        super().__init__()
        self.bias = bias
        self.max_value = max_value
        self.relu = nn.ReLU(inplace) if leak is None else nn.LeakyReLU(leak, inplace)

    def forward(self, x):
        x = self.relu(x) + self.bias
        if self.max_value is not None:
            x = torch.clamp_max_(x, self.max_value)
        return x
