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
    RandomAffineTfm(scale, [0.9, 1.1]),
    RandomAffineTfm(rotate, 10),
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

            loss = self.objective(img)
            loss.backward()

            # print(img_param.noise.grad.abs().max(), img_param.noise.grad.abs().mean(),img_param.noise.grad.std())

            # Apply transforms to the gradient (normalize, blur, etc.)
            # with torch.no_grad():
            #     img_param.noise.grad.data = img_param.noise.grad.data / (img_param.noise.grad.data.std() + 1e-1)
            #     input_img.grad.data = ReducingGaussianBlur(3, 3, 5)(input_img.grad.data)
            # print(img_param.noise.grad.abs().max(), img_param.noise.grad.abs().mean(),img_param.noise.grad.std())


            self.optim.step()
            self.optim.zero_grad()

            if verbose and i in thresh:
                print(i, loss.item())
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
    """Defines an Objective which OptVis will optimise. The
    Objective class should have a callable function which
    should return the loss associated with the forward pass.
    This class has the same functionality as Lucid: objectives
    can be summed, multiplied by scalars, negated or subtracted.
    """
    def __init__(self, objective_function, name=None):
        """
        Parameters:
        objective_function: function that returns the loss of the network.
        name (str): name of the objective. Used for display. (optional)
        """
        self.objective_function = objective_function
        self.name = name

    def __call__(self, x):
        return self.objective_function(x)

    @property
    def cls_name(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.cls_name}" if self.name is None else self.name

    def __add__(self, other):
        if isinstance(other, (int,float)):
            name = " + ".join([self.__repr__(), other.__repr__()])
            return Objective(lambda x: other + self(x), name=name)
        elif isinstance(other, Objective):
            name = " + ".join([self.__repr__(), other.__repr__()])
            return Objective(lambda x: other(x) + self(x), name=name)
        else:
            raise TypeError(f"Can't add value of type {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int,float)):
            name = f"{other}*{self.__repr__()}"
            return Objective(lambda x: other * self(x), name=name)
        else:
            raise TypeError(f"Can't add value of type {type(other)}")

    def __sub__(self, other):
        return self + (-1*other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self.__mul__(-1.)

class LayerObjective(Objective):
    """Generate an Objective from a particular layer of a network.
    Supports the layer indexing that interpret provides as well as
    options for selecting the channel or neuron of the layer.

    Parameters:
    model (nn.Module): PyTorch model.
    layer (str or int): the layer to optimise.
    channel (int): the channel to optimise. (optional)
    neuron (int): the neuron to optimise. (optional)
    shortcut (bool): Whether to attempt to shortcut the network's
        computation. Only works for Sequential type models.
    """
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

    def objective_function(self, x):
        "Apply the input to the network and set the loss."
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

        return self.loss

    def __repr__(self):
        msg = f"{self.cls_name}: {self.layer}"
        if self.channel is not None:
            msg += f":{self.channel}"
        if self.neuron is not None:
            msg += f":{self.neuron}"
        if self.channel is None and self.neuron is not None and self.model[self.layer].weight.size(0)==1000:
            msg += f"  {imagenet_labels[self.neuron]}"
        return msg
