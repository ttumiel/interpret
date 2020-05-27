import pytest, torch
import numpy as np
from PIL import Image

from interpret.utils import norm, denorm

def test_norm():
    data = (np.random.random((32,32,3))*255).astype('uint8')
    img = Image.fromarray(data)
    data = norm(img, mean=(data/255).mean((0,1)), std=(data/255).std((0,1)))

    assert data.mean().item() == pytest.approx(0., abs=1e-6)
    assert data.std().item() == pytest.approx(1., 1e-3)

def test_denorm():
    data = torch.randn(1,3,50,50)
    denorm_data = denorm(data, image=False)

    assert denorm_data.max() <= 255
    assert denorm_data.min() >= 0
    assert denorm_data.shape == (50,50,3)

def test_norm_denorm():
    img = Image.fromarray(np.random.random((32,32,3)), 'RGB')

    data = norm(img)
    denorm_data = denorm(data, image=False)

    assert np.isclose(np.array(img), denorm_data).all()
