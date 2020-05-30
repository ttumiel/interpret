import pytest
import torch
from PIL import Image
import numpy as np

from interpret.transforms import scale, translate, shear, rotate, RandomTransform, _random_params

@pytest.fixture(scope='module')
def tensor():
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.arange(4).view(2,2).float()[None, None].to(d)


def test_translate(tensor):
    trans_x = translate(1, 0)(tensor)
    out = torch.stack([tensor[...,0], tensor[...,0]]).permute((1,2,3,0))
    assert torch.all(trans_x == out)

    trans_y = translate(0, 1)(tensor)
    out = torch.stack([tensor[...,1,:], tensor[...,1,:]]).permute((1,2,0,3))
    assert torch.all(trans_y == out)

def test_rotate(tensor):
    rot_45 = rotate(45)(tensor)
    t = tensor.squeeze()
    out = torch.tensor([
        [(t[0,0]+t[0,1])/2, (t[0,1]+t[1,1])/2],
        [(t[0,0]+t[1,0])/2, (t[1,0]+t[1,1])/2],
    ]).to(rot_45.device)
    assert torch.all(rot_45 == out)

def test_scale(tensor):
    scale_2 = scale(2)(tensor)

    im = Image.fromarray(tensor.cpu().squeeze().numpy()).resize((4,4), resample=Image.BILINEAR)
    out = torch.tensor(np.array(im)[1:3,1:3]).to(tensor.device)
    assert torch.all(out == scale_2)

def test_shear(tensor):
    shear_1 = shear(1)(tensor)
    out = tensor.clone()
    out[...,0,1] = (out[...,0,0]+out[...,0,1])/2
    out[...,1,0] = (out[...,1,0]+out[...,1,1])/2
    assert torch.all(out == shear_1)

def test_random_transform():
    def raise_call():
        raise RuntimeError('This transform should not have run but did.')
    t = RandomTransform(raise_call, p=0)
    for _ in range(10):
        t(None)

def test_random_params():
    for _ in range(10):
        out = _random_params(1,2)
        assert -1 <= out[0] <= 1
        assert -2 <= out[1] <= 2

        out = _random_params([0,2],[-1,4])
        assert 0 <= out[0] <= 2
        assert -1 <= out[1] <= 4

