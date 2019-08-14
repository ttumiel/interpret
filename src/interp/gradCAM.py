import torch.nn.functional as F
from ..hooks import *

def gradcam(model, img, im_class, layer_num=0, heatmap_thresh:int=16, image:bool=True):
    m = model.eval()
    cl = int(im_class)
    xb = img

    with hook_output(m[layer_num]) as hook_a:
        with hook_output(m[layer_num], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cl)].backward()
    acts  = hook_a.stored[0].cpu()
    if (acts.shape[-1]*acts.shape[-2]) >= heatmap_thresh:
        grad = hook_g.stored[0][0].cpu()
        grad_chan = grad.mean(1).mean(1)
        mult = F.relu(((acts*grad_chan[...,None,None])).sum(0))
        if image:
            xb_im = Image.fromarray(denorm(xb[0]))
            _,ax = plt.subplots()
            sz = list(xb_im.size)
            ax.imshow(xb_im)
            ax.imshow(mult, alpha=0.4, extent=(0,*sz[::-1],0), interpolation='bilinear', cmap='magma')
        return mult
