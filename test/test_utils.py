import pytest, torch
import numpy as np
from interpret.utils import norm, denorm
from interpret.data import random_shapes

def test_norm():
    img = random_shapes(size=30, shape='circle', min_size=10, max_size=50,
                        coord_limits=None, background='color', color=True, number=5)[0]
    data = norm(img, mean=np.array(img).mean((0,1))/255, std=np.array(img).std((0,1))/255)

    assert data.mean().item() == pytest.approx(0., abs=1e-6)
    assert data.std().item() == pytest.approx(1., 1e-3)

def test_denorm():
    data = torch.randn(1,3,50,50)
    denorm_data = denorm(data, image=False)

    assert denorm_data.max() <= 255
    assert denorm_data.min() >= 0
    assert denorm_data.shape == (50,50,3)

def test_norm_denorm():
    img = random_shapes(size=50, shape='circle', min_size=10, max_size=50,
                        coord_limits=None, background='color', color=True, number=5)[0]

    data = norm(img)
    denorm_data = denorm(data, image=False)

    assert np.isclose(np.array(img), denorm_data).all()
