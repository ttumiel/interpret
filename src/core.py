"""Core utilities for this project"""

import torch

def get(s,i):
    if isinstance(i, int):
        return list(s.children())[i]
    elif isinstance(i, str):
        layers = i.split("/")
        l = s
        for layer in layers:
            l = getattr(l, layer)
        return l

torch.nn.Module.__getitem__ = get
