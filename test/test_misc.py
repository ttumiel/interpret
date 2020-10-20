import torch
from torch import nn

from interpret.misc import *

class DeterministicNetwork(nn.Module):
    "A fake module that returns what you give it sequentially."
    def __init__(self, data, device):
        super().__init__()
        self.data = data
        self.count = 0
        self.device = device

    def forward(self, x):
        bs = x.size(0)
        index = self.data[self.count:self.count + bs]
        self.count += bs
        return index.to(self.device)

def test_top_losses(network, dataloader, conv_layer, n_classes, device):
    n = 4
    loss_fn = nn.CrossEntropyLoss()
    p,y,l,i = top_losses(network, dataloader, loss_fn)
    assert len(p) == len(y) == len(l) == len(i) == len(dataloader.dataset)
    assert torch.all(l.sort(descending=True).values == l)

    plot_top_losses((p,y,l,i), dataloader)
    plot_top_losses((p,y,l,i), dataloader, network=network, n=n, gradcam=True, layer=conv_layer)

    tgts = torch.cat([t for _,t in dataloader]).float()
    fake_net = DeterministicNetwork(tgts, device)
    p,y,l,i = top_losses(fake_net, dataloader, nn.MSELoss())
    assert torch.all(l == 0) and torch.all(p==y)


def test_confusion_matrix(network, dataloader, n_classes, device):
    cm = confusion_matrix(network, dataloader, n_classes)
    assert cm.shape[0] == n_classes and cm.shape[1] == n_classes
    assert cm.max() <= len(dataloader.dataset) and cm.min() >= 0
    assert cm.sum() == len(dataloader.dataset)

    plot_confusion_matrix(cm)

    tgts = torch.cat([t for _,t in dataloader]).float()
    fake_net = DeterministicNetwork(tgts, device)
    cm = confusion_matrix(fake_net, dataloader, n_classes)
    assert cm.diagonal().sum() == len(dataloader.dataset)


def test_dataset_examples(network, dataloader, device, conv_layer, channel):
    idxs = dataset_examples(network, dataloader, conv_layer, channel, device)
    assert len(idxs) == len(dataloader.dataset)
    assert idxs.min()>=0 and idxs.max()<len(dataloader.dataset)

    plot_dataset_examples(idxs, dataloader)


def test_loss_landscape(network, dataloader, device):
    X,Y,Z = loss_landscape(network, dataloader, device=device)
    assert X.shape == Y.shape == Z.shape

    plot_loss_landscape((X,Y,Z))
    plot_loss_landscape((X,Y,Z), mode='contour')
