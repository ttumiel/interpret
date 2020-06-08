import torchvision, torch
import random, pytest

@pytest.fixture(scope='session')
def network(n_classes, device):
    return torchvision.models.resnet18(pretrained=False, num_classes=n_classes).to(device)

@pytest.fixture(scope='session')
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope='session')
def dataloader():
    ds = FakeDataset(n=10, ins=torch.randn, outs=torch.zeros(10, dtype=torch.long))
    return torch.utils.data.DataLoader(ds, 5)

@pytest.fixture(scope='session')
def n_classes():
    return 5

@pytest.fixture(scope='session')
def conv_layer():
    return 'layer3/1/conv2'

@pytest.fixture(scope='session')
def linear_layer():
    return 'fc'

@pytest.fixture(scope='session')
def imsize():
    return 64

@pytest.fixture(scope='session')
def n_steps():
    return (30,)

@pytest.fixture
def neuron(n_classes):
    return random.randrange(n_classes)

@pytest.fixture
def channel():
    return random.randrange(256)

@pytest.fixture
def channels():
    return (random.randrange(256), random.randrange(256))


class FakeDataset():
    """A fake dataset for testing.

    Parameters:
        n (int): the number of items in the ds.
        ins (function): the function generating the input.
        outs (list): a list of outputs of len n.
    """
    def __init__(self, n, ins, outs):
        self.n = n
        self.outs = outs
        self.ins = ins(n,3,64,64)
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return self.ins[idx], self.outs[idx]
