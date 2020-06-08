import torch
from torch import nn

from ..utils import find_all
# from ..hooks import hook_output
from .attribute import Attribute
from ..models.layers import Lambda
from interpret.override import ModuleOverride

# def guided_relu(module, input, output):
#     "Apply relu to forward and backward pass."
#     return tuple(torch.relu(inp) for inp in output) if isinstance(input, tuple) else torch.relu(output)

# def deconv_relu(module, input, output):
#     "Apply deconvnet relu to forward and backward pass."
#     return tuple(torch.relu(out) for out in output) if isinstance(output, tuple) else torch.relu(output)

class GuidedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[grad_input < 0] = 0
        return grad_input

class DeconvnetReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input[grad_input < 0] = 0
        return grad_input

# TODO: Change from intrusive autograd functions to hooks if possible
class GuidedBackProp(Attribute):
    """Implements Guided Backpropagation [1]. This algorithms
    creates a saliency map using the gradient of the network
    while masking out both negative values present in the
    forward pass as well as the backward pass.

    [1] - https://arxiv.org/pdf/1412.6806.pdf
    """
    def __init__(self, model, input_img, target_class, deconvnet=False):
        m = model.eval()
        self.input_data = input_img

        if self.input_data.grad is not None:
            self.input_data.grad.fill_(0)

        relu_override = DeconvnetReLU.apply if deconvnet else GuidedReLU.apply
        with ModuleOverride(m, nn.ReLU, Lambda(relu_override)):
            # hooks = [l.register_backward_hook(guided_relu) for l in relu_modules]

            # with hook_output(m) as h:
            loss = m(input_img)[0, target_class]

            loss.backward()

            self.data = input_img.grad.detach().clone().squeeze()
            # input_img.grad.fill_(0)

            # for h in hooks:
            #     h.remove()
