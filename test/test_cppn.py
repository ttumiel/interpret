import torch

from interpret import Objective, CPPNParam, OptVis

from . import assert_loss_decreases


@Objective
def xor_objective(a):
    a = torch.sigmoid(a)
    out = (torch.square(a[..., 0, 0]) + torch.square(a[..., -1, -1]) +
           torch.square(1.0-a[..., -1, 0]) + torch.square(1.0-a[..., 0, -1]))
    return -torch.sum(out)


def test_cppn_loss(network, conv_layer, channel):
    cppn = CPPNParam(64)
    optvis = OptVis.from_layer(network, conv_layer, channel=channel)
    assert_loss_decreases(optvis, img_param=cppn, thresh=(50,), lr=0.01, transform=False)


def test_cppn_xor(network, conv_layer, channel):
    optvis = OptVis(network, xor_objective)

    for _ in range(3): # 3 attempts
        cppn = CPPNParam(64)
        optvis.vis(cppn, thresh=(50,), lr=0.01, transform=False, verbose=False)

        if check_channel(torch.sigmoid(cppn())):
            return

    raise RuntimeError('CPPN failed xor after 3 tries.')


def check_channel(im):
    return bool(
        torch.all(im[..., 0, 0] > 0.85) and
        torch.all(im[..., 0, -1] < 0.15) and
        torch.all(im[..., -1, 0] < 0.15) and
        torch.all(im[..., 1, 1] > 0.85)
    )
