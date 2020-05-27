import pytest

from interpret import OptVis
from interpret.hooks import Hook

from . import assert_loss_decreases


def test_neuron(network, imsize, neuron, n_steps):
    optvis = OptVis.from_layer(network, layer="classifier/6", neuron=neuron)
    assert_loss_decreases(optvis, thresh=n_steps)

def test_neuron_fail(network, channel):
    with pytest.raises(AssertionError):
        optvis = OptVis.from_layer(network, layer="classifier/6", channel=channel, neuron=6)
        optvis.vis(verbose=False)

def test_channel(network, imsize, channel, n_steps):
    optvis = OptVis.from_layer(network, layer="features/18", channel=channel, neuron=6)
    assert_loss_decreases(optvis, thresh=n_steps)

def test_shortcut(network, imsize, channel, n_steps):
    def was_called(m,i,o):
        raise Exception

    with Hook(network[1], was_called):
        optvis = OptVis.from_layer(network, layer="features/18", channel=channel, neuron=6, shortcut=True)
        assert_loss_decreases(optvis, thresh=n_steps)
