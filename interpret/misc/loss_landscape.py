import copy
import math
import torch
import numpy as np
from torch import nn
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

from interpret.misc import validate


def loss_landscape(network, dataloader, dir1=None, dir2=None, dir1_bound=(-1,1), dir2_bound=(-1,1),
                   proc_fn=np.log, clip=(-3,10), step=0.1, device=None):
    """Generate the losses of a network using a grid of weights,
    originating with the network's current weights, and moving
    in a grid pattern between (-dir1, dir1) and (-dir2, dir2).

    To use a custom direction, set dir1/dir2 as a state_dict with
    the same keys as the network. For example, this can be set
    as the same network trained from a different random seed,
    a random initialization, or anything in between. Note that these
    must be directions from the supplied network state dict
    i.e the difference between the target state dict and the
    network state dict.

    Parameters:
        network (nn.Module): The (trained) network to generate
            the loss landscape of.
        dataloader (DataLoader): The pytorch dataloader to generate
            the loss values from. Usually you can reduce the size of the
            dataset to ~5% of the original size without changing the
            landscape visualisation much.
        dir1 and dir2 (state_dict): the direction to move across the loss
            landscape. Defaults to random normal tensors.
        dir1_bound and dir2_bound ((float, float)): the bounds to move in
            the direction of dir1 and dir2. Usually unnecessary to change.
        proc_fn (callable): function to post process the losses.
            Use np.log to smooth large spikes in the landscape.
            Use np.sqrt for less smoothing.
            Use None to apply no post-processing.
        clip (tuple): Clip the min and max loss values. Use None to ignore.
        step (float): the fidelity of the loss landscape generation.
        device (torch.device): torch device for validation.

    Returns (tuple):
        The X,Y,Z values for the loss landscape.
    """
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    trained_sd = copy.deepcopy(network.state_dict())

    losses = []
    x_pts = math.ceil((dir1_bound[1]-dir1_bound[0])/step+1)
    y_pts = math.ceil((dir2_bound[1]-dir2_bound[0])/step+1)
    total = x_pts*y_pts
    for sd in tqdm(get_state_dicts(trained_sd, dir1, dir2, step=step,
                                   dir1_bound=dir1_bound, dir2_bound=dir2_bound),
                   total=total, desc='Generating losses'):
        network.load_state_dict(sd)
        p,y = validate(network, dataloader, device=device)
        loss = nn.CrossEntropyLoss()(p,y.squeeze())
        losses.append(loss.cpu().detach().numpy())

    # Restore original state
    network.load_state_dict(trained_sd)

    X = np.arange(dir1_bound[0],dir1_bound[1]+step,step)
    Y = np.arange(dir2_bound[0],dir2_bound[1]+step,step)
    X,Y = np.meshgrid(X,Y)

    Z = np.array(losses).reshape((x_pts, y_pts)).T
    if proc_fn is not None: Z = proc_fn(Z)
    if clip is not None: Z = np.clip(Z, *clip)

    return X,Y,Z

def plot_loss_landscape(XYZ=None, network=None, dl=None, mode='surface', elevation=40, angle=45,
                       levels=None, figsize=(10,10), **pyplot_kwds):
    """Plot the loss landscape as either a surface or contour plot.
    Use the outputs of `loss_landscape` directly or provide the network
    to create the landscape from.

    Parameters:
        XYZ (tuple): The outputs of `loss_landscape`.
        network (nn.Module): Trained pytorch network to get the landscape of.
        mode (str): 'surface' or 'contour'
        elevation (int): elevation to view the surface plot from.
        angle (int): angle to view the surface plot from.
        levels (list): the levels of the contour plot to show. Defaults to
            the min and max losses, with a step of 0.5.
        figsize (float, float): pyplot figure size.
        **pyplot_kwds: kwargs passed on to surface or contour plot functions.

    Returns (Axes):
        pyplot axes with the image.
    """
    assert (XYZ is not None) ^ (network is not None and dl is not None), "Either XYZ or network must be set."
    X,Y,Z = loss_landscape(network, dl) if network is not None else XYZ

    if mode == 'contour':
        levels = levels if levels is not None else np.arange(np.min(Z),np.max(Z),0.5)
        f,ax = plt.subplots(figsize=figsize)
        c = ax.contour(X,Y,Z, levels=levels, **pyplot_kwds)
        ax.clabel(c, inline=2, fontsize=8)
        f.colorbar(c)
    elif mode == 'surface':
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none', **pyplot_kwds)
        ax.view_init(elevation, angle)
    else: raise ValueError('mode `{}` not recognised'.format(mode))
    return ax

def get_state_dicts(original, dir1=None, dir2=None, step=0.1, dir1_bound=(-1,1), dir2_bound=(-1,1)):
    "Generate state dicts by interpolating the original with 2 directions."
    dir1 = dir1 or get_rand_dir(original)
    dir2 = dir2 or get_rand_dir(original)
    for d1 in np.arange(dir1_bound[0], dir1_bound[1]+step, step):
        for d2 in np.arange(dir2_bound[0], dir2_bound[1]+step, step):
            yield {k:v+dir1[k]*d1+dir2[k]*d2 for k,v in original.items()}

def normalize_direction(d, w):
    "Make the norm of the weight and direction the same."
    return d.mul_(w.norm()/(d.norm() + 1e-10))

def get_rand_dir(sd):
    "Get random tensors in the shape of all of the elements in the state dict."
    weight_dir = {}
    for k,v in sd.items():
        if v.dtype is torch.float and v.ndim>1:
            weight_dir[k] = normalize_direction(torch.randn(v.size()).to(v.device)-v, v)
        else: # ignore batch norm and integer layers
            weight_dir[k] = torch.zeros_like(v)
    return weight_dir
