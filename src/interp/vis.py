"Visualise models"

from ..hooks import Hook
import torch
import core
from PIL import Image

class CutModel():
    "Class to visualise particular layers by optimisation"
    def __init__(self, model, layer, channel):
        self.model,self.layer,self.channel=model,layer,channel
        self.active = False
        for p in self.model.parameters():
            p.requires_grad_(False)

    def __call__(self, x):
        def activation_fn(module,input,output):
            self.loss = -torch.mean(output[:,self.channel])
            self.active = True

        with Hook(self.model[self.layer], activation_fn, detach=False):
            for i,m in enumerate(self.model.children()):
                x = m(x)
                if self.active:
                    self.active = False
                    break
        return x

def random_im(size=64):
    "Create a random 'image' that is normalized according to the network"
    im = torch.rand((3,size,size))*30 + 160
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    im /= 255
    im -= mean[..., None, None]
    im /= std[..., None, None]
    im.unsqueeze_(0)
    im.requires_grad_(True)
    return im

def denorm(im):
    im = im.detach().clone().cpu().squeeze()
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    im *= std[..., None, None]
    im += mean[..., None, None]
    im *= 254
    im += 0.5
    im = im.permute(1,2,0).numpy()
    im[im>255]=255
    im[im<0]=0

    im = Image.fromarray(im.round().astype('uint8'))
    return im
