def assert_loss_decreases(optvis, train=True, **kwargs):
    if train: optvis.vis(**kwargs, verbose=False)
    assert bool(optvis.losses[0] > optvis.losses[-1])
