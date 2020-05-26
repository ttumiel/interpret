"Visualise models"

import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from ..core import *
from ..utils import *
from ..transforms import *
from ..utils import denorm
from .objective import *
from .param import *

VIS_TFMS = [
    RandomAffineTfm(scale, [0.9, 1.1]),
    RandomAffineTfm(rotate, 10),
]

class OptVis():
    """
    Class to visualise particular layers by optimisation. Visualisation
    follows the procedure outlined by Olah et al. [1] and
    implemented in Lucid [2].

    Parameters:
        model (nn.Module): PyTorch model.
        objective (Objective): The objective that the network will optimise.
            See factory methods from_layer.
        tfms (list): list of transformations to potentially apply to image.
        grad_tfms (list): list of transformations to apply to the gradient of the image.
        optim (torch.optim): PyTorch optimisation function.
        shortcut (bool): Attempt to shorten the computation by iterating through
            the layers until the objective is reached as opposed to calling the
            entire network. Only works on Sequential-like models.

    [1] - https://distill.pub/2017/feature-visualization/
    [2] - https://github.com/tensorflow/lucid
    """

    def __init__(self, model, objective, transforms=None, optim=torch.optim.Adam, shortcut=False, device=None, grad_tfms=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' if device is None else device
        self.model = model.to(self.device).eval()
        self.objective = objective
        self.active = False
        self.tfms = transforms if transforms is not None else VIS_TFMS.copy()
        self.grad_tfms = grad_tfms
        self.optim_fn = optim
        self.shortcut = shortcut
        self.upsample = True
        print(f"Optimising for {objective}")

    def vis(self, img_param=None, thresh=(500,), transform=True, lr=0.05, wd=0., verbose=True):
        """
        Generate a visualisation by optimisation of an input. Updates img_param in-place.

        Parameters:
            img_param: object that parameterises the input noise.
            thresh (tuple): thresholds at which to display the generated image.
                Only displayed if verbose==True. Input optimised for max(thresh) iters.
            transform (bool): Whether to transform the input image using self.tfms.
            lr (float): learning rate for optimisation.
            wd (float): weight decay for self.optim_fn.
            verbose (bool): display input on thresholds.
        """
        if verbose:
            try:
                from IPython.display import display
            except ImportError:
                raise ValueError("Can't use verbose if not in IPython notebook.")

        if img_param is None:
            img_param = ImageParam(128)

        img_param.to(self.device)

        if img_param.size < 224 and self.upsample:
            self.tfms.append(torch.nn.Upsample(size=224, mode='bilinear', align_corners=True))
            self.upsample = False

        transforms = torchvision.transforms.Compose(self.tfms)

        freeze(self.model.eval(), bn=True)
        self.optim = self.optim_fn(img_param.parameters(), lr=lr, weight_decay=wd)
        for i in tqdm(range(1,max(thresh)+1)):
            img = img_param()

            if transform:
                img = transforms(img)

            loss = self.objective(img)
            loss.backward()

            # Apply transforms to the gradient (normalize, blur, etc.)
            if self.grad_tfms is not None:
                with torch.no_grad():
                    img_param.noise.grad.data = self.grad_tfms(img_param.noise.grad.data)

            self.optim.step()
            self.optim.zero_grad()

            if verbose and i in thresh:
                print(i, loss.item())
                display(denorm(img_param()))

        return img_param

    @classmethod
    def from_layer(cls, model, layer, channel=None, neuron=None, shortcut=False, **kwargs):
        "Factory method to create OptVis from a LayerObjective. See respective classes for docs."
        if ":" in layer:
            layer, channel = layer.split(":")
            channel = int(channel)
        obj = LayerObjective(model, layer, channel, neuron=neuron, shortcut=shortcut)
        return cls(model, obj, **kwargs)

    @classmethod
    def from_dream(cls, model, layer, **kwargs):
        "Factory method for deepdream objective"
        return cls(model, DeepDreamObjective(model, layer), **kwargs)
