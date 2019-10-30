import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..hooks import *
from .. import core
from ..utils import denorm

def gradcam(model, img, im_class, layer=0, heatmap_thresh=16, image=True,
            show_im=True, ax=None, colorbar=False, cmap='magma', alpha=0.4):
    """Generates a Grad-CAM attribution map for convolutional neural networks.

    Parameters:
    model: PyTorch model.
    img (torch.Tensor): input tensor fed into the network for attribution.
    im_class (int): the class that the network is attributing on.
    layer (int): the layer the network is using for the attribution. See [1].
    heatmap_thresh (int): prevents heatmaps from being created when the
        feature map is less than 2x2 pixels.
    image (bool): show the heatmap as an image
    show_im (bool): show the denormalised input image overlaid on the heatmap.
    ax: axes on which to plot image.
    colorbar (bool): show a colorbar.
    cmap: matplotlib colourmap.
    alpha (float): transparency value alpha for heatmap.

    Returns:
    The Grad-CAM heatmap (torch.Tensor)
    """
    m = model.eval()
    cl = int(im_class)
    xb = img

    with hook_output(m[layer]) as hook_a:
        with hook_output(m[layer], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cl)].backward()
    acts  = hook_a.stored[0].cpu()
    if (acts.shape[-1]*acts.shape[-2]) >= heatmap_thresh:
        grad = hook_g.stored[0][0].cpu()
        grad_chan = grad.mean(1).mean(1)
        mult = F.relu(((acts*grad_chan[...,None,None])).sum(0))
        if image:
            xb_im = Image.fromarray(denorm(xb[0], image=False))
            if ax is None: _,ax = plt.subplots()
            sz = list(xb_im.size)
            if show_im:
                ax.imshow(xb_im)
            im = ax.imshow(mult, alpha=alpha, extent=(0,*sz,0), interpolation='bilinear', cmap=cmap)
            if colorbar:
                ax.figure.colorbar(im, ax=ax)
        return mult

def gradcam_from_examples(learn, n_examples, layer, figsize=(10,10), show_overlay=False, cmap='magma'):
    "Utility method to generate a collage of attribution maps from a list of examples."
    c = learn.data.dataset.c
    ax = plt.subplots(n_examples,c + (2 if show_overlay else 1),figsize=figsize)[1]
    for row in range(n_examples):
        img, label = learn.val_data.dataset[np.random.randint(len(learn.val_data.dataset))]
        img = img.to(learn.device)
        ax[row][0].imshow(denorm(img))
        if c>1:
            pred = learn.predict((img[None], label))[0].argmax(1)
        else:
            pred = learn.predict((img[None], label))[0].round()
        pred_str = learn.val_data.dataset.decode_label(pred.item())
        label_str = learn.val_data.dataset.decode_label(label.item())
        ax[row][0].set_title(f"Input Image.\nPrediction: {pred_str}.\nLabel: {label_str}.")
        if show_overlay:
            gradcam(learn.model, img[None], pred if pred<c else 0, layer=layer, show_im=True, ax=ax[row][1], cmap=cmap)
            ax[row][1].set_title(f'Overlay of Predicted Class {int(pred.item())}')
        for class_label in range(c):
            gradcam(learn.model, img[None], class_label, layer=layer, show_im=False,
                    ax=ax[row][class_label+(2 if show_overlay else 1)], cmap=cmap)
            if c > 1:
                ax[0][class_label + (2 if show_overlay else 1)].set_title(f"Looking for: {learn.val_data.dataset.decode_label(class_label)}")
            else:
                ax[0][class_label + (2 if show_overlay else 1)].set_title(f"Prediction")
    ax=ax.flatten()
    for i in range(len(ax)):
        ax[i].set_axis_off()
