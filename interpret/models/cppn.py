import torch
from torch import nn

from interpret.models import Lambda

__all__ = ['CPPN', 'composite_act', 'ReLUNormalized']

def composite_act(x):
    x = torch.atan(x)
    return torch.cat([x/0.67, (x*x)/0.6], 1)

CompositeAct = Lambda(composite_act)
ArcTan = Lambda(torch.atan)
ReLUNormalized = Lambda(lambda x: torch.relu(x).sub(0.4).div(0.58))

class CPPN(nn.Sequential):
    def __init__(self, cout=3, num_hidden=24, num_layers=8,
                 act=CompositeAct, normalize=False, ks=1):
        m = act(torch.randn(1,1,1,1)).shape[1]
        net = [nn.Conv2d(2, num_hidden, 1), act]
        for _ in range(num_layers):
            net.append(nn.Conv2d(num_hidden*m, num_hidden, ks, padding=ks//2))
            if normalize:
                net.append(nn.InstanceNorm2d(num_hidden))
            net.append(act)

        net.append(nn.Conv2d(num_hidden*m, cout, ks, padding=ks//2))
        super().__init__(*net)
