import torch.nn.functional as F
from ..hooks import *
from .. import core
from PIL import Image
from ..utils import denorm
import matplotlib.pyplot as plt

def gradcam(model, img, im_class, layer=0, heatmap_thresh=16, image=True, show_im=True, ax=None):
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
            if show_im: ax.imshow(xb_im)
            ax.imshow(mult, alpha=0.4, extent=(0,*sz[::-1],0), interpolation='bilinear', cmap='magma')
        return mult
