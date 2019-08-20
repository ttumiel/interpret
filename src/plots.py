"""Plot images"""

from ..transforms import denorm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_image(tensor, normalize=False, ax=None):
    img = denorm(tensor) if normalize else tensor
    plt.imshow(img) if ax is not None else ax.imshow(img)

def show_images(batched_tensor, normalize=False, figsize=(5,5)):
    r = np.ceil(np.sqrt(batched_tensor.size(0)))
    [show_image(batched_tensor[i],normalize,ax) for i,ax in enumerate(plt.subplots(r,r,figsize=figsize)[1].flatten())]
