"Visualisation Objectives"

import torch
from torch import nn
from functools import partial

from interpret.hooks import Hook
from interpret.core import *
from interpret.imagenet import imagenet_stats, imagenet_labels

__all__ = ['Objective', 'LayerObjective', 'DeepDreamObjective']

class Objective():
    """Defines an Objective which OptVis will optimise. The
    Objective class should have a callable function which
    should return the loss associated with the forward pass.
    This class has the same functionality as Lucid: objectives
    can be summed, multiplied by scalars, negated or subtracted.

    To create a new Objective class, you can either subclass from
    this class so that you can add state or you can decorate a
    function with @Objective to turn it into an Objective.
    """
    def __init__(self, objective_function=None, name=None):
        """
        Parameters:
            objective_function: function that returns the loss of the network.
            name (str): name of the objective. Used for display. (optional)
        """
        if objective_function is not None:
            self.objective_function = objective_function
        self.name = name

    def __call__(self, x):
        return self.objective_function(x)

    @property
    def cls_name(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"{self.cls_name}" if not hasattr(self, 'name') or self.name is None else self.name

    def __add__(self, other):
        if isinstance(other, (int,float)):
            name = " + ".join([repr(self), repr(other)])
            return Objective(lambda x: other + self(x), name=name)
        elif isinstance(other, Objective):
            name = " + ".join([repr(self), repr(other)])
            return Objective(lambda x: other(x) + self(x), name=name)
        else:
            raise TypeError(f"Can't add value of type {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int,float)):
            name = f"{other}*{repr(self)}"
            return Objective(lambda x: other * self(x), name=name)
        else:
            raise TypeError(f"Can't add value of type {type(other)}")

    def __sub__(self, other):
        return self + (-1*other)

    def __rsub__(self, other):
        return -1*self + other

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return -1. * self

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
        batchwise (bool): Calculate the loss for each element in the batch.
    """
    def __init__(self, model, layer, channel=None, neuron=None, shortcut=False, batchwise=False):
        self.model = model
        self.layer = layer
        self.channel = channel
        self.neuron = neuron
        self.shortcut = shortcut
        self.batchwise = batchwise
        if self.shortcut:
            self.active = False

        try:
            self.model[layer]
        except AttributeError:
            raise ValueError(f"Can't find layer {layer}. Use 'get_layer_names' to print all layer names.")

    def objective_function(self, x):
        "Apply the input to the network and set the loss."
        def layer_hook(module, input, output):
            rank = len(output.shape)
            c = self.channel or slice(None)
            n = self.neuron or slice(None)
            offset = ((0 if self.channel is None else self.channel)
                      + (0 if self.neuron is None else self.neuron*2))
            dims = list(range(1,rank - offset)) if self.batchwise else []

            if rank == 4:
                self.loss = -torch.mean(output[:, c, n, n], dim=dims)
            elif rank == 2:
                self.loss = -torch.mean(output[:, n], dim=dims)
                assert self.channel is None, f"Channel is unused for layer {self.layer}"

            # Set flag for shortcutting the computation
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

class DeepDreamObjective(Objective):
    """Deep Dream objective from [1]. Maximises all features of
    a particular layer

    [1] - https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html
    """
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer

    def objective_function(self, x):
        def layer_hook(module, input, output):
            self.loss = -torch.mean(output**2)

        with Hook(self.model[self.layer], layer_hook, detach=False, clone=True):
            x = self.model(x)

        return self.loss


@partial(Objective, name='TotalVariation')
def total_variation(x):
    """Calculates the total variation of neighbouring pixels of an input image

    Usually used as a penalty to reduce noise. A coefficient of around 1e-5 works well.
    """
        width_sum = torch.sum(torch.abs(x[...,1:] - x[..., :-1]))
        height_sum = torch.sum(torch.abs(x[...,1:,:] - x[...,:-1,:]))
        return width_sum + height_sum
