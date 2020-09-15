"""Core utilities for this project"""

from torch import nn

__all__ = ['freeze', 'unfreeze']

def get_module_item(s,i):
    if isinstance(i, int):
        if isinstance(s, nn.Sequential):
            return _orig_seq_get(s, i)
        else:
            return list(s.children())[i]
    elif isinstance(i, str):
        layers = i.split("/")
        l = s
        for layer in layers:
            l = getattr(l, layer)
        return l

def set_module_item(self, m, new_m):
    if "/" in m:
        root, name = m.rsplit('/', maxsplit=1)
        root_module = self[root]
        setattr(root_module, name, new_m)
    else:
        setattr(self, m, new_m)

def freeze(s,bn=False):
    def inner(m):
        if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.requires_grad_(False)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad_(False)
    s.apply(inner)

def unfreeze(s):
    for p in s.parameters():
        p.requires_grad_(True)

_orig_seq_get = nn.Sequential.__getitem__
nn.Sequential.__getitem__ = get_module_item
nn.Module.__getitem__ = get_module_item
nn.Sequential.__setitem__ = set_module_item
nn.Module.__setitem__ = set_module_item
nn.Module.freeze = freeze
nn.Module.unfreeze = unfreeze
