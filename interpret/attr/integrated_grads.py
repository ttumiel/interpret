import torch
from . import Attribute, Gradient

class IntegratedGradients(Attribute):
    """
    Integrated gradients [1] - a method to generate attribution maps by integrating
    over the gradient of an image across the path from a baseline image, to
    the actual image. The baseline image is chosen so that it results in neutral
    predictions such as zero vector or noise.

    Parameters:
        model:
        input_image:
        target_class:
        steps:
        baseline:

    [1] - https://arxiv.org/pdf/1703.01365.pdf
    """
    def __init__(self, model, input_image, target_class, steps=50, baseline=None):
        if baseline is None:
            baseline = input_image.detach().clone().requires_grad_(False).fill_(0)

        grads = []
        for a in range(steps):
            inp = (baseline + (input_image - baseline)*a/steps).detach().clone().requires_grad_(True)
            y = model(inp)
            y[:, target_class].backward()
            grads.append(inp.grad)
            # input_image.grad.fill_(0)

        self.data = torch.stack(grads).mean(0)
        self.input_data = input_image

        # sum([Gradient(model, (baseline + (input_image - baseline)*a/steps), target_class) for a in range(steps)])
