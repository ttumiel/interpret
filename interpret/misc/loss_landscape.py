import copy
import math
import torch
import numpy as np
from torch import nn
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

from interpret.misc import validate

def loss_landscape(network, dataloader, loss_fn=nn.CrossEntropyLoss(), dir1=None, dir2=None,
                   dir1_bound=(-1,1,20), dir2_bound=(-1,1,20), device=None):
    """Generate the losses of a network using a grid of weights,
    originating with the network's current weights, and moving
    in a grid pattern between (-dir1, dir1) and (-dir2, dir2).

    To use a custom direction, set dir1/dir2 as a state_dict with
    the same keys as the network. For example, this can be set
    as the same network trained from a different random seed,
    a random initialization, or anything in between. Note that these
    are directions from the supplied `network` state dict
    i.e the directions are added to the network's current state
    dict.

    For details, see: Li et al, Visualizing the Loss Landscape of
    Neural Nets. https://arxiv.org/abs/1712.09913

    Parameters:
        network (nn.Module): The (trained) network to generate
            the loss landscape of.
        dataloader (DataLoader): The pytorch dataloader to generate
            the loss values from. Usually you can reduce the size of the
            dataset to ~5% of the original size without changing the
            landscape visualisation much.
        loss_fn (callable): The loss function to evaluate.
        dir1 and dir2 (state_dict): the direction to move across the loss
            landscape. Defaults to random normal tensors.
        dir1_bound and dir2_bound ((float, float, int)): (start, stop, num_steps)
            the bounds to move in the direction of dir1 and dir2.
        device (torch.device): torch device for validation.

    Returns (tuple):
        The X,Y,Z values for the loss landscape.
    """
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    trained_sd = copy.deepcopy(network.state_dict())

    losses = []
    x_pts = dir1_bound[2]
    y_pts = dir2_bound[2]
    total = x_pts*y_pts
    for sd in tqdm(get_state_dicts(trained_sd, dir1, dir2,
                                   dir1_bound=dir1_bound, dir2_bound=dir2_bound),
                   total=total, desc='Generating losses'):
        network.load_state_dict(sd)
        p,y = validate(network, dataloader, device=device)
        loss = loss_fn(p,y.squeeze())
        losses.append(loss.cpu().detach().numpy())

    # Restore original state
    network.load_state_dict(trained_sd)

    X = np.linspace(*dir1_bound)
    Y = np.linspace(*dir2_bound)
    X,Y = np.meshgrid(X,Y)
    Z = np.array(losses).reshape((x_pts, y_pts)).T

    return X,Y,Z

def plot_loss_landscape(XYZ=None, network=None, dl=None, mode='surface', proc_fn=np.log,
                        clip=(-3,10), elevation=40, angle=45, levels=None, figsize=(10,10),
                        label=False, cmap='viridis', **pyplot_kwds):
    """Plot the loss landscape as either a surface or contour plot.
    Use the outputs of `loss_landscape` directly or provide the network
    and dataloader to create the landscape from.

    Parameters:
        XYZ (tuple): The outputs of `loss_landscape`.
        network (nn.Module): Trained pytorch network to get the landscape of.
        dl (DataLoader): Pytorch dataloader for landscape generation.
        mode (str): 'surface' or 'contour'
        proc_fn (callable): function to post process the losses.
            Use np.log to smooth large spikes in the landscape.
            Use np.sqrt for less smoothing.
            Use None to apply no post-processing.
        clip (tuple): Clip the min and max loss values. Use None to ignore.
        elevation (int): elevation to view the surface plot from.
        angle (int): angle to view the surface plot from.
        levels (list): the levels of the contour plot to show. Defaults to
            the min and max losses, with a step of 0.5.
        figsize (float, float): pyplot figure size.
        label (bool): Add axis labels to the plot.
        cmap: pyplot colourmap.
        **pyplot_kwds: kwargs passed on to surface or contour plot functions.

    Returns (Axes):
        pyplot axes with the image.
    """
    assert (XYZ is not None) ^ (network is not None and dl is not None), "Either XYZ or network must be set."
    X,Y,Z = loss_landscape(network, dl) if network is not None else XYZ

    # Post-process loss values
    if proc_fn is not None: Z = proc_fn(Z)
    if clip is not None: Z = np.clip(Z, *clip)

    if mode == 'contour':
        levels = levels if levels is not None else np.arange(np.min(Z),np.max(Z),0.5)
        f,ax = plt.subplots(figsize=figsize)
        c = ax.contour(X,Y,Z, levels=levels, cmap=cmap, **pyplot_kwds)
        ax.clabel(c, inline=2, fontsize=8)
        f.colorbar(c)
    elif mode == 'surface':
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cmap, edgecolor='none', **pyplot_kwds)
        ax.view_init(elevation, angle)
        if label: ax.set_zlabel('Loss')
    else: raise ValueError('mode `{}` not recognised'.format(mode))
    if label:
        ax.set_xlabel('Dir 1')
        ax.set_ylabel('Dir 2')
    return ax

def get_state_dicts(original, dir1=None, dir2=None, dir1_bound=(-1,1,20), dir2_bound=(-1,1,20)):
    "Generate state dicts by interpolating the original with 2 directions."
    dir1 = dir1 or get_rand_dir(original)
    dir2 = dir2 or get_rand_dir(original)
    for d1 in np.linspace(*dir1_bound):
        for d2 in np.linspace(*dir2_bound):
            yield {k:v+dir1[k]*d1+dir2[k]*d2 for k,v in original.items()}

def normalize_direction(d, w):
    "Make the norm of the weight and direction the same for every filter."
    for fd, fw in zip(d, w):
        fd.mul_(fw.norm()/(fd.norm() + 1e-10))
    return d

def get_rand_dir(sd):
    """Get random tensors in the shape of all of the elements in the state dict.
    Each weight is normalized across each filter.
    """
    weight_dir = {}
    for k,v in sd.items():
        if v.dtype is torch.float and v.ndim>1:
            weight_dir[k] = normalize_direction(torch.randn(v.size()).to(v.device), v)
        else: # ignore batch norm and integer layers
            weight_dir[k] = torch.zeros_like(v)
    return weight_dir

def get_dir_from_target(sd, target_sd):
    "Get a direction vector starting at sd and ending at target_sd"
    weight_dir = {}
    for k,v in sd.items():
        weight_dir[k] = target_sd[k] - v
    return weight_dir
