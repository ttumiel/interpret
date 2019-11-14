import torch
from torch import nn

from interpret.attr import GuidedBackProp

def test_guidedbackprop():
    inp = torch.randn(1,50, requires_grad=True)
    l1 = nn.Linear(50, 1)
    model = nn.Sequential(nn.ReLU(), l1)

    attr = GuidedBackProp(model, inp, target_class=0)

    relu_inp = torch.relu(inp).detach().clone().requires_grad_(True)
    l1(relu_inp).backward()

    manual_guide = relu_inp.grad.clone()
    manual_guide[inp<0] = 0
    manual_guide[manual_guide<0] = 0
    assert (manual_guide == inp.grad).all()

def test_deconvnet():
    inp = torch.randn(1,50, requires_grad=True)
    l1 = nn.Linear(50, 1)
    model = nn.Sequential(nn.ReLU(), l1)

    attr = GuidedBackProp(model, inp, target_class=0, deconvnet=True)

    relu_inp = torch.relu(inp).detach().clone().requires_grad_(True)
    l1(relu_inp).backward()

    manual_guide = relu_inp.grad.clone()
    manual_guide[manual_guide<0] = 0

    assert (manual_guide == inp.grad).all()
