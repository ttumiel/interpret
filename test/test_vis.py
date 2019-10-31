import pytest
from interpret import OptVis, ImageParam, denorm
import torchvision, torch

def test_neuron():
    network = torchvision.models.vgg11(pretrained=False).to('cuda' if torch.cuda.is_available() else 'cpu')
    layer = 'classifier/6'
    neuron = 5
    optvis = OptVis.from_layer(network, layer=layer, neuron=neuron)
    img_param = ImageParam(224, fft=True, decorrelate=True)
    optvis.vis(img_param, thresh=(10,), transform=True, lr=0.05, wd=0.9, verbose=False)

def test_channel():
    network = torchvision.models.vgg11(pretrained=False).to('cuda' if torch.cuda.is_available() else 'cpu')
    layer = 'classifier/6'
    channel = 23
    optvis = OptVis.from_layer(network, layer=layer, channel=channel)
    img_param = ImageParam(224, fft=True, decorrelate=True)
    optvis.vis(img_param, thresh=(10,), transform=True, lr=0.05, wd=0.9, verbose=False)
