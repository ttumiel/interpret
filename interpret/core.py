"""Core utilities for this project"""

from torch import nn

def get(s,i):
    if isinstance(i, int):
        return list(s.children())[i]
    elif isinstance(i, str):
        layers = i.split("/")
        l = s
        for layer in layers:
            l = getattr(l, layer)
        return l

def freeze(s,bn=False):
    for p in s.parameters():
        if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            p.requires_grad_(False)

def unfreeze(s):
    for p in s.parameters():
        p.requires_grad_(True)

nn.Module.__getitem__ = get
nn.Module.freeze = freeze
nn.Module.unfreeze = unfreeze
