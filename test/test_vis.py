import pytest

from interpret import OptVis
from interpret.hooks import Hook

from . import assert_loss_decreases


def test_neuron(network, imsize, linear_layer, neuron, n_steps):
    optvis = OptVis.from_layer(network, linear_layer, neuron=neuron)
    assert_loss_decreases(optvis, thresh=n_steps)

def test_neuron_fail(network, linear_layer, channel, n_classes):
    with pytest.raises(AssertionError):
        optvis = OptVis.from_layer(network, linear_layer, channel=channel, neuron=n_classes-1)
        optvis.vis(verbose=False)

def test_channel(network, imsize, conv_layer, channel, n_steps):
    optvis = OptVis.from_layer(network, conv_layer, channel=channel, neuron=6)
    assert_loss_decreases(optvis, thresh=n_steps)

def test_shortcut(network, imsize, conv_layer, channel, n_steps):
    def was_called(m,i,o):
        raise Exception

    with Hook(network['layer4/0/conv1'], was_called):
        optvis = OptVis.from_layer(network, conv_layer, channel=channel, neuron=6, shortcut=True)
        assert_loss_decreases(optvis, thresh=n_steps)
