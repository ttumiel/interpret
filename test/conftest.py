import torchvision, torch
import random, pytest

@pytest.fixture(scope='session')
def network():
    return torchvision.models.vgg11(pretrained=False).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

@pytest.fixture(scope='session')
def conv_layer():
    return 'features/18'

@pytest.fixture(scope='session')
def linear_layer():
    return 'classifier/6'

@pytest.fixture(scope='session')
def imsize():
    return 64

@pytest.fixture(scope='session')
def n_steps():
    return (10,)

@pytest.fixture
def neuron():
    return random.randrange(1000)

@pytest.fixture
def channel():
    return random.randrange(512)

@pytest.fixture
def channels():
    return (random.randrange(512), random.randrange(512))
