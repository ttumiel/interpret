import pytest

from interpret import OptVis, LayerObjective, ImageFile

from . import assert_loss_decreases


def test_objective_operators(network, conv_layer, channels, n_steps):
    objective1 = LayerObjective(network, conv_layer, channel=channels[0])
    objective2 = LayerObjective(network, conv_layer, channel=channels[1])

    # Sum
    optvis = OptVis(network, objective1+objective2)
    assert_loss_decreases(optvis, thresh=n_steps)

    # Sub
    optvis = OptVis(network, objective1-1)
    assert_loss_decreases(optvis, thresh=n_steps)

    optvis = OptVis(network, 1-objective1)
    assert_loss_decreases(optvis, thresh=n_steps)

    # Neg
    optvis = OptVis(network, -objective1)
    assert_loss_decreases(optvis, thresh=n_steps)

    # Product
    with pytest.raises(TypeError):
        optvis = OptVis(network, objective1*objective2)
        optvis.vis(thresh=(1,), verbose=False)

    optvis = OptVis(network, 2*objective1)
    assert_loss_decreases(optvis, thresh=n_steps)

    optvis = OptVis(network, objective1*2)
    assert_loss_decreases(optvis, thresh=n_steps)


def test_dream(network, imsize, conv_layer, n_steps):
    optvis = OptVis.from_dream(network, conv_layer)
    img_param = ImageFile("test/fixtures/dog.jpg", imsize)
    optvis.vis(img_param, thresh=n_steps, transform=False, verbose=False)
    assert_loss_decreases(optvis, train=False)
