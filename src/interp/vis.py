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

tfms = torchvision.transforms.Compose([
    torchvision.transforms.RandomApply([Blur(2)], p=0.02),
    # torchvision.transforms.RandomApply([RandomTfm(rotate, 15)], p=0.1),
    # torchvision.transforms.RandomApply([RandomTfm(scale, [0.9, 1.1])], p=0.1),
])


class CutModel():
    "Class to visualise particular layers by optimisation"

    def __init__(self, model, layer, channel, tfms=tfms, optim=torch.optim.Adam):
        self.model, self.layer, self.channel = model, layer, channel
        self.active = False
        self.tfms = tfms
        self.optim_fn = optim
        print(f"Optimising for layer {layer}, channel {channel}")
        for p in self.model.parameters():
            p.requires_grad_(False)

    def __call__(self, x):
        def activation_fn(module, input, output):
            self.loss = -torch.mean(output[:, self.channel])
            self.active = True

        with Hook(self.model[self.layer], activation_fn, detach=False):
            for i, m in enumerate(self.model.children()):
                x = m(x)
                if self.active:
                    self.active = False
                    break
        return x

    def run(self, input_img, transform=False, iters=50):
        self.optim = self.optim_fn([input_img], lr=0.05, weight_decay=1e-6)
        for i in range(iters):
            self(input_img)
            self.loss.backward()

            if i % 10 == 0:
                print(i, self.loss.item())
                display(zoom(denorm(input_img)))

            if transform and i % 6 == 0:
                with torch.no_grad():
                    input_img = self.tfms(input_img)
                input_img.requires_grad_(True)

            self.optim.step()
            self.optim.zero_grad()
            input_img.data.clamp_(-2, 2)


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


def denorm(im, decorrelate=False):
    im = im.detach().clone().cpu().squeeze()
    mean, std = imagenet_stats
    mean, std = torch.tensor(mean), torch.tensor(std)
    if decorrelate: im = _linear_decorelate_color(im)

    im *= std[..., None, None]
    im += mean[..., None, None]
    im *= 254
    im += 0.5
    im = im.permute(1, 2, 0).numpy()
    im[im > 255] = 255
    im[im < 0] = 0

    im = Image.fromarray(im.round().astype('uint8'))
    return im
