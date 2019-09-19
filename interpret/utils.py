import torch
from pathlib import Path
from PIL import Image

from .imagenet import imagenet_stats

Path.ls = lambda c: list(c.iterdir())

def zoom(im, zoom=2):
    return im.transform((int(im.size[0]*zoom), int(im.size[1]*zoom)), Image.EXTENT, (0, 0, im.size[0], im.size[1]))

def denorm(im, mean=imagenet_stats[0], std=imagenet_stats[1], image=True):
    "Denormalize an image"
    if isinstance(im, torch.Tensor):
        im = im.detach().clone().cpu().squeeze()

    mean, std = torch.tensor(mean), torch.tensor(std)

    im *= std[..., None, None]
    im += mean[..., None, None]
    im *= 254
    im += 0.5
    im = im.permute(1, 2, 0).numpy()
    im[im > 255] = 255
    im[im < 0] = 0

    im = im.round().astype('uint8')
    if not image: return im
    return Image.fromarray(im)


def norm(im, mean=imagenet_stats[0], std=imagenet_stats[1]):
    "Normalize an image"
    mean, std = torch.tensor(mean), torch.tensor(std)
    im /= 255
    im -= mean[..., None, None]
    im /= std[..., None, None]
    im.unsqueeze_(0)
    im.requires_grad_(True)
    return im