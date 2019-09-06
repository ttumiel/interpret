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
    """Multiply input by sqrt of emperical (ImageNet) color correlation matrix.

    If you interpret t's innermost dimension as describing colors in a
    decorrelated version of the color space (which is a very natural way to
    describe colors -- see discussion in Feature Visualization article) the way
    to map back to normal colors is multiply the square root of your color
    correlations.
    """
    # check that inner dimension is 3?
    t_flat = t.squeeze().view([3, -1])
    color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt
    t_flat = torch.tensor(color_correlation_normalized) @ t_flat
    t = t_flat.view(t.size())
    return t

def fourier_image(size=64, noise_scale=0.01):
    noise=noise_scale*torch.randn([3,size,size,2])
    tfm_noise = torch.fft(noise, 3, normalized=True)
    noise = torch.irfft(tfm_noise, 3, onesided=False)
    return noise.unsqueeze_(0).clone().detach().requires_grad_(True)
