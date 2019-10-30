"Visualise models"

import numpy as np
import torch
import torchvision
from PIL import Image

from ..hooks import Hook
from ..core import *
from ..imagenet import imagenet_stats, imagenet_labels
from ..utils import *
from ..transforms import *
from ..utils import denorm
from .param import *

VIS_TFMS = torchvision.transforms.Compose([
    RandomEvery([
                                        GaussianBlur(3, 3, 0.5),
                                        RandomAffineTfm(rotate, 5),
                    RandomAffineTfm(scale, [0.98, 1.05])
                ], p=0.5)
])

class OptVis():
    """
    Class to visualise particular layers by optimisation.

    Parameters:
    model (nn.Module): PyTorch model.
    objective (Objective): The objective that the network will optimise.
        See factory methods from_layer.
    tfms (list): list of transformations to potentially apply to image.
    optim (torch.optim): PyTorch optimisation function.
    shortcut (bool): Attempt to shortten the computation by iterating through
        the layers until the objective is reached as opposed to calling the
        entire network. Only works on Sequential-like models.
    """

    def __init__(self, model, objective, tfms=VIS_TFMS, optim=torch.optim.Adam, shortcut=False):
        self.model = model
        self.objective = objective
        self.active = False
        self.tfms = tfms
        self.optim_fn = optim
        self.shortcut = shortcut
        print(f"Optimising for {objective}")
        self.model.eval()

    def vis(self, img_param, thresh=(500,), transform=True, lr=0.05, wd=0., verbose=True):
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

        freeze(self.model, bn=True)
        self.optim = self.optim_fn(img_param.parameters(), lr=lr, weight_decay=wd)
        for i in range(max(thresh)+1):
            img = img_param()

            if transform:
                img = self.tfms(img)

            self.objective(img)
            self.objective.loss.backward()

            # print(img_param.noise.grad.abs().max(), img_param.noise.grad.abs().mean(),img_param.noise.grad.std())

            # Apply transforms to the gradient (normalize, blur, etc.)
            # with torch.no_grad():
            #     img_param.noise.grad.data = img_param.noise.grad.data / (img_param.noise.grad.data.std() + 1e-1)
            #     input_img.grad.data = ReducingGaussianBlur(3, 3, 5)(input_img.grad.data)
            # print(img_param.noise.grad.abs().max(), img_param.noise.grad.abs().mean(),img_param.noise.grad.std())


            self.optim.step()
            self.optim.zero_grad()

            if verbose and i in thresh:
                print(i, self.objective.loss.item())
                display(zoom(denorm(img), 2))

            # self.optim.param_groups[0]['params'][0] = img_obj['optimise']

    @classmethod
    # layer and channel... How to make this extensible into layer,
    # channel and neuron?? Separate classes feels wasteful
    def from_layer(cls, model, layer, channel=None, neuron=None, shortcut=False, **kwargs):
        "Factory method to create OptVis from a LayerObjective. See respective classes for docs."
        if ":" in layer:
            layer, channel = layer.split(":")
            channel = int(channel)
        obj = LayerObjective(model, layer, channel, neuron=neuron, shortcut=shortcut)
        return cls(model, obj, **kwargs)

class Objective():
    # the objective class should have a callable function which
    # should define the function on which to optimise.
    # define __add__, __sub__, neg, mult, div, ...
    def __init__(self):
        pass

    def __call__(self, module, input, output):
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.name}"

class LayerObjective(Objective):
    def __init__(self, model, layer, channel=None, neuron=None, shortcut=False):
        self.model = model
        self.layer = layer
        self.channel = channel
        self.neuron = neuron
        self.shortcut = shortcut
        if self.shortcut:
            self.active = False

        try:
            self.model[layer]
        except:
            raise ValueError(f"Can't find layer {layer}. Use 'get_layer_names' to print all layer names.")

    # Feels belaboured? Change to separate classes for channel, layer, neuron
    def __call__(self, x):
        def layer_hook(module, input, output):
            if self.neuron is None:
                if self.channel is None:
                    self.loss = -torch.mean(output)
                else:
                    self.loss = -torch.mean(output[:, self.channel])
            else:
                if isinstance(module, nn.Conv2d):
                    # TODO: Check if channel is None and handle
                    self.loss = -torch.mean(output[:, self.channel, self.neuron])
                elif isinstance(module, nn.Linear):
                    self.loss = -torch.mean(output[:, self.neuron])
            self.active = True

        with Hook(self.model[self.layer], layer_hook, detach=False, clone=True):
            if self.shortcut:
                for i, m in enumerate(self.model.children()):
                    x = m(x)
                    if self.active:
                        self.active = False
                        break
            else:
                x = self.model(x)

    def __repr__(self):
        msg = f"{self.name}: {self.layer}"
        if self.channel is not None:
            msg += f":{self.channel}"
        if self.neuron is not None:
            msg += f":{self.neuron}"
        if self.channel is None and self.neuron is not None and self.model[self.layer].weight.size(0)==1000:
            msg += f"  {imagenet_labels[self.neuron]}"
        return msg

# class NeuronObjective(Objective):
#     def __init__(self, model, layer):
#         self.model = model
#         self.layer = layer

#     def __call__(self, *inputs, **kwargs):
#         if self.neuron is None:
#             self.loss = -torch.mean(output[:, self.channel])
#         else:
#             if isinstance(module, nn.Conv2d):
#                 self.loss = -torch.mean(output[:, self.channel, self.neuron])
#         self.active = True
