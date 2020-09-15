import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from interpret.hooks import *
from interpret import core
from interpret.utils import denorm
from interpret.attr import Attribute

class Gradcam(Attribute):
    """Generates a Grad-CAM attribution map for convolutional neural networks.

    Parameters:
        model: PyTorch model.
        img (torch.Tensor): input tensor fed into the network for attribution.
        im_class (int): the class that the network is attributing on.
        layer (int): the layer the network is using for the attribution. See [1].
        heatmap_thresh (int): prevents heatmaps from being created when the
            feature map is less than 2x2 pixels.

    Returns:
    The Grad-CAM heatmap (torch.Tensor)

    References:
    [1] - Grad-CAM: Visual Explanations from Deep Networks via
          Gradient-based Localization. https://arxiv.org/abs/1610.02391
    """
    def __init__(self, model, img, im_class, layer=0, heatmap_thresh=16):
        self.input_data = img
        m = model.eval()
        cl = int(im_class)
        xb = img
        m[layer].requires_grad_(True)

        with hook_output(m[layer]) as hook_a:
            with hook_output(m[layer], grad=True) as hook_g:
                preds = m(xb)
                preds[0,int(cl)].backward()
                acts = hook_a.stored[0].cpu()
                grad = hook_g.stored[0][0].cpu()
        if (acts.shape[-1]*acts.shape[-2]) >= heatmap_thresh:
            grad_chan = grad.mean(1).mean(1)
            self.data = F.relu(((acts*grad_chan[...,None,None])).sum(0))
        else:
            raise ValueError("Image not large enough to create a heatmap. Increase "
                            "size of image or move the layer further down into the "
                            "network")
