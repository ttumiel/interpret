"""Plot images"""

from .utils import denorm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import math

def show_image(tensor, normalize=False, ax=None, **kwargs):
    img = denorm(tensor, **kwargs) if normalize else np.array(tensor.clone())
    plt.imshow(img) if ax is None else ax.imshow(img)

def show_images(batched_tensor, normalize=False, figsize=(5,5), axis=False, labels=None, **kwargs):
    r = math.ceil(math.sqrt(batched_tensor.size(0)))
    axes = plt.subplots(r,r,figsize=figsize)[1].flatten()
    for i,ax in enumerate(axes):
        if i<batched_tensor.size(0):
            show_image(batched_tensor[i],normalize,ax)
            if labels is not None: ax.set_title(f'{labels[i]}')
        if not axis: ax.set_axis_off()

def plot(y, x=None, title=None, ax=None, x_lb=None, y_lb=None):
    if ax is None:
        plt.plot(y) if x is None else plt.plot(y, x)
        plt.title(title)
        plt.xlabel(x_lb)
        plt.ylabel(y_lb)
    else:
        ax.plot(y) if x is None else ax.plot(y, x)
        ax.set_title(title)
        ax.set_xlabel(x_lb)
        ax.set_ylabel(y_lb)
