from interpret import OptVis, ImageParam, ImageFile

from . import assert_loss_decreases


def test_image_params(network, imsize, conv_layer, channel, n_steps):
    optvis = OptVis.from_layer(network, conv_layer, channel=channel)

    # with fft and decorrelate
    img_param = ImageParam(imsize, fft=True, decorrelate=True)
    assert_loss_decreases(optvis, img_param=img_param, thresh=n_steps)

    # with fft and without decorrelate
    img_param = ImageParam(imsize, fft=True, decorrelate=True)
    assert_loss_decreases(optvis, img_param=img_param, thresh=n_steps)

    # without fft and with decorrelate
    img_param = ImageParam(imsize, fft=True, decorrelate=True)
    assert_loss_decreases(optvis, img_param=img_param, thresh=n_steps)

    # without fft and decorrelate
    img_param = ImageParam(imsize, fft=False, decorrelate=False)
    assert_loss_decreases(optvis, img_param=img_param, thresh=n_steps)
