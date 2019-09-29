"Visualise models"

from ..hooks import Hook
import torch
from .. import core
from PIL import Image
from ..imagenet import imagenet_stats, imagenet_labels
from ..utils import *
from IPython.display import display
import torchvision
from ..transforms import *
import numpy as np
from ..utils import denorm

VIS_TFMS = torchvision.transforms.Compose([
    RandomEvery([
                                        GaussianBlur(3, 3, 0.5),
                                        RandomAffineTfm(rotate, 5),
                    RandomAffineTfm(scale, [0.98, 1.05])
                ], p=0.5)
])


class OptVis():
    "Class to visualise particular layers by optimisation"

    def __init__(self, model, objective, tfms=VIS_TFMS, optim=torch.optim.Adam, shortcut=False):
        self.model = model
        self.objective = objective
        self.active = False
        self.tfms = tfms
        self.optim_fn = optim
        self.shortcut = shortcut
        print(f"Optimising for {objective}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def __call__(self, x):
        def activation_fn(module, input, output):
            if self.neuron is None:
                self.loss = -torch.mean(output[:, self.channel])
            else:
                if isinstance(module, nn.Conv2d):
                    self.loss = -torch.mean(output[:, self.channel, self.neuron])
            self.active = True

        with Hook(self.model[self.layer], activation_fn, detach=False):
            for i, m in enumerate(self.model.children()):
                x = m(x)
                if self.active:
                    self.active = False
                    break
        return x

    def vis(self, input_img, transform=False, iters=50, decorrelate=False, lr=0.05, wd=0.):
        self.optim = self.optim_fn([input_img], lr=lr, weight_decay=wd)
        for i in range(iters):
            self.objective(input_img)
            self.objective.loss.backward() # Maybe change to just objective.backward?

            if i % 100 == 0: # add if verbose/print_every
                print(i, self.objective.loss.item())
                display(zoom(denorm(input_img), 2))

            self.optim.step()
            self.optim.zero_grad()
            # input_img.data.clamp_(-2, 2)

            if transform:
                with torch.no_grad():
                    input_img = self.tfms(input_img)

            # if decorrelate: # Move to image generation
            #     input_img = _linear_decorelate_color(input_img).detach()

            input_img = input_img.requires_grad_(True)
            self.optim = self.optim_fn([input_img], lr=lr, weight_decay=wd)

        return input_img

    @classmethod
    # layer and channel... How to make this extensible into layer,
    # channel and neuron?? Separate classes feels wasteful
    def from_layer(cls, model, layer, neuron=None, shortcut=False, **kwargs):
        channel = None
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
    def __init__(self, model, layer, channel, neuron=None, shortcut=False):
        self.model = model
        self.layer = layer
        self.channel = channel
        self.neuron = neuron
        self.shortcut = shortcut
        if self.shortcut:
            self.active = False

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
                    # Check if channel is None and handle
                    self.loss = -torch.mean(output[:, self.channel, self.neuron])
                elif isinstance(module, nn.Linear):
                    self.loss = -torch.mean(output[:, self.neuron])
            self.active = True

        with Hook(self.model[self.layer], layer_hook, detach=False):
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
        if self.channel is None and self.neuron is not None:
            msg += f"  {imagenet_labels[self.neuron]}"
        return msg

def random_im(size=64):
    "Create a random 'image' that is normalized according to the network"
    im = torch.rand((3, size, size))*30 + 160
    mean, std = imagenet_stats
    mean, std = torch.tensor(mean), torch.tensor(std)
    im /= 255
    im -= mean[..., None, None]
    im /= std[..., None, None]
    im.unsqueeze_(0)
    im.requires_grad_(True)
    return im

color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                        [0.27, 0.00, -0.05],
                                        [0.27, -0.09, 0.03]]).astype("float32")


max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
color_mean = [0.48, 0.46, 0.41]

def _linear_decorelate_color(t):
    """Multiply input by sqrt of empirical (ImageNet) color correlation matrix.

    If you interpret t's innermost dimension as describing colors in a
    decorrelated version of the color space (which is a very natural way to
    describe colors -- see discussion in Feature Visualization article) the way
    to map back to normal colors is multiply the square root of your color
    correlations.
    """
    # check that inner dimension is 3?
    t_flat = t.squeeze().view([3, -1])
    color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
    t_flat = torch.tensor(color_correlation_normalized).t() @ t_flat # should this be transposed??
    t = t_flat.view(t.size())
    return t

def fourier_image(size=64, noise_scale=0.01, decorrelate=False):
    noise=noise_scale*torch.randn([3,size,size,2])
    tfm_noise = torch.fft(noise, 3, normalized=True)
    noise = torch.irfft(tfm_noise, 3, onesided=False)
    if decorrelate:
        noise = _linear_decorelate_color(noise)
    return noise.unsqueeze_(0).clone().detach().requires_grad_(True)
