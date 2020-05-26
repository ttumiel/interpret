import torchvision, torch, pytest
import random

from interpret import OptVis, ImageParam, ImageFile
from interpret.hooks import Hook

@pytest.fixture
def network():
    return torchvision.models.vgg11(pretrained=False).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

@pytest.fixture
def imsize():
    return 64

@pytest.fixture
def neuron():
    return random.randrange(1000)

@pytest.fixture
def channel():
    return random.randrange(512)

def test_neuron(network, imsize, neuron):
    optvis = OptVis.from_layer(network, layer="classifier/6", neuron=neuron)
    img_param = ImageParam(imsize, fft=True, decorrelate=True)
    optvis.vis(img_param, thresh=(10,), transform=True, lr=0.05, verbose=False)

def test_neuron_fail(network, channel):
    with pytest.raises(AssertionError):
        optvis = OptVis.from_layer(network, layer="classifier/6", channel=channel, neuron=6)
        optvis.vis(verbose=False)

def test_channel(network, imsize, channel):
    optvis = OptVis.from_layer(network, layer="features/18", channel=channel, neuron=6)
    optvis.vis(thresh=(10,), verbose=False)

def test_shortcut(network, imsize, channel):
    def was_called(m,i,o):
        raise Exception

    with Hook(network[1], was_called):
        optvis = OptVis.from_layer(network, layer="features/18", channel=channel, neuron=6, shortcut=True)
        optvis.vis(thresh=(10,), verbose=False)

def test_image_params(network, imsize, channel):
    optvis = OptVis.from_layer(network, layer="features/18", channel=channel, neuron=6)
    optvis.vis(thresh=(1,), verbose=False)
    init_loss = optvis.objective.loss

    # with fft and decorrelate
    img_param = ImageParam(imsize, fft=True, decorrelate=True)
    optvis.vis(img_param, thresh=(10,), verbose=False)
    assert bool(optvis.objective.loss < init_loss)

    # with fft and without decorrelate
    img_param = ImageParam(imsize, fft=True, decorrelate=True)
    optvis.vis(img_param, thresh=(10,), verbose=False)
    assert bool(optvis.objective.loss < init_loss)

    # without fft and with decorrelate
    img_param = ImageParam(imsize, fft=True, decorrelate=True)
    optvis.vis(img_param, thresh=(10,), verbose=False)
    assert bool(optvis.objective.loss < init_loss)

    # without fft and decorrelate
    img_param = ImageParam(imsize, fft=False, decorrelate=False)
    optvis.vis(img_param, thresh=(10,), verbose=False)
    assert bool(optvis.objective.loss < init_loss)
