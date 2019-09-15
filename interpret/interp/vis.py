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

VIS_TFMS = torchvision.transforms.Compose([
    # torchvision.transforms.RandomApply([GaussianBlur(3, 3, 0.5)], p=0.001),
    torchvision.transforms.RandomApply([
                                        GaussianBlur(3, 3, 0.5),
                                        RandomTfm(rotate, 5),
                                        RandomTfm(scale, [0.95, 1.05])
                                        ], p=0.5),
    # torchvision.transforms.RandomApply([RandomTfm(scale, [0.95, 1.05])], p=0.005),
])


class CutModel():
    "Class to visualise particular layers by optimisation"

    def __init__(self, model, layer, channel, tfms=VIS_TFMS, optim=torch.optim.Adam, neuron=None):
        self.model, self.layer, self.channel = model, layer, channel
        self.active = False
        self.tfms = tfms
        self.optim_fn = optim
        self.neuron = neuron
        print(f"Optimising for layer {layer}, channel {channel}")
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

    def vis(self, input_img, transform=False, iters=50, decorrelate=False, lr=0.05, wd=1e-3):
        self.optim = self.optim_fn([input_img], lr=lr, weight_decay=wd)
        for i in range(iters):
            self(input_img)
            self.loss.backward()

            if i % 100 == 0:
                print(i, self.loss.item())
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


def denorm(im):
    im = im.detach().clone().cpu().squeeze()
    mean, std = imagenet_stats
    mean, std = torch.tensor(mean), torch.tensor(std)

    im *= std[..., None, None]
    im += mean[..., None, None]
    im *= 254
    im += 0.5
    im = im.permute(1, 2, 0).numpy()
    im[im > 255] = 255
    im[im < 0] = 0

    im = Image.fromarray(im.round().astype('uint8'))
    return im


def norm(im):
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
