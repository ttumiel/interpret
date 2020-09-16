import torch
from pathlib import Path
from PIL import Image
import numpy as np

from interpret.imagenet import imagenet_stats

__all__ = ['zoom', 'denorm', 'norm', 'get_layer_names']

Path.ls = lambda c: list(c.iterdir())

def zoom(im, zoom=2):
    return im.transform((int(im.size[0]*zoom), int(im.size[1]*zoom)), Image.EXTENT, (0, 0, im.size[0], im.size[1]), resample=Image.BILINEAR)

def denorm(im, mean=imagenet_stats[0], std=imagenet_stats[1], image=True):
    "Denormalize an image"
    if isinstance(im, torch.Tensor):
        im = im.detach().clone().cpu().squeeze()
    mean, std = torch.tensor(mean), torch.tensor(std)

    im *= std[..., None, None]
    im += mean[..., None, None]
    im *= 255
    im = im.permute(1, 2, 0).clamp(0,255).numpy()

    im = im.round().astype('uint8')
    if not image: return im
    return Image.fromarray(im)


def norm(im, input_range=(0,255), mean=imagenet_stats[0], std=imagenet_stats[1], unsqueeze=True, grad=True):
    "Normalize an image"
    if isinstance(im, Image.Image):
        im = torch.tensor(np.asarray(im).copy()).permute(2,0,1).float()
    elif isinstance(im, np.ndarray):
        im = torch.tensor(im).float()
        size = im.size()
        assert len(size)==3 or len(size)==4, "Image has wrong number of dimensions."
        assert size[0]==3 or size[0]==1, "Image has invalid channel number. Should be 1 or 3."

    mean, std = torch.tensor(mean, device=im.device), torch.tensor(std, device=im.device)
    im = im + input_range[0]
    im = im / input_range[1]
    im = im - mean[..., None, None]
    im = im / std[..., None, None]
    if unsqueeze: im.unsqueeze_(0)
    if grad: im.requires_grad_(True)
    return im

def get_layer_names(module, display=True, names=None, upper_name='', _title=True):
    """Recursively show a network's named layers

    Parameters:
        module (nn.Module): The pytorch module to display the submodule names.
        display (bool): print out a table of the names with attempts at sizes.

    Returns (List):
        List of submodule names.
    """
    fstr = "{:^40} | {:^18} | {:^10} | {:^10}"
    if names is None:
        names = []

    if _title:
        print(fstr.format("Layer", "Class Name", "Input Size", "Output Size"))
        print(f"{'-'*40} | {'-'*18} | {'-'*10} | {'-'*10}")

    for name, m in module._modules.items():
        if m is not None:
            if display:
                print(fstr.format(
                    upper_name+name,
                    m.__class__.__name__,
                    m.weight.size(1) if hasattr(m, 'weight') and len(m.weight.shape)>1 else '-',
                    m.weight.size(0) if hasattr(m, 'weight') else '-'))

            names.append(upper_name+name)
            get_layer_names(m, display, names, upper_name+name+"/", False)

    return names

def find_all(model, module_type, path=False, _upper_name=""):
    """Find all Modules of a particular type in `model`.

    Parameters:
        model (nn.Module): the pytorch module to search through.
        module_type (type): the instance type to match for. i.e.
            matches `isinstance(m, module_type)`.
        path (bool): return the path name that returns the found module.

    Returns (List, [List]):
        A list of the modules matchine module_type. Optionally includes
        the paths from the root module to the found module.
    """
    matches = []
    pathnames = []
    for pathname, m in model._modules.items():
        if isinstance(m, module_type):
            matches += [m]
            if path:
                pathnames += [_upper_name + pathname]
        else:
            next_find = find_all(m, module_type, path=path, _upper_name=_upper_name+pathname + '/')
            if path:
                matches += next_find[0]
                pathnames += next_find[1]
            else:
                matches += next_find
    if path:
        return matches, pathnames
    return matches
