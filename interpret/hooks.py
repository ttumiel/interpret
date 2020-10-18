import torch

class Hook:
    """
    Create a hook on `m` with `hook_func`.

    Adapted from fastai: https://github.com/fastai/fastai/
    """
    def __init__(self, m, hook_func, is_forward=True, detach=True, clone=False):
        self.hook_func = hook_func
        self.detach = detach
        self.clone = clone
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module, input, output):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input  = (o.detach() for o in input ) if is_listy(input ) else input.detach()
            output = (o.detach() for o in output) if is_listy(output) else output.detach()

        if self.clone:
            input  = (o.clone() for o in input ) if is_listy(input ) else input.clone()
            output = (o.clone() for o in output) if is_listy(output) else output.clone()

        return self.hook_func(self, module, input, output)

    def remove(self):
        "Remove the hook from the model."
        self.hook.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

    def __del__(self):
        self.remove()


def _hook_output(h,m,i,o):
    h.stored = o if isinstance(o,torch.Tensor) else o if is_listy(o) else list(o)

def hook_output(model, detach=True, grad=False, clone=False):
    return Hook(model, _hook_output, detach=detach, is_forward=not grad, clone=clone)

def _hook_input(h,m,i,o):
    h.stored = i if isinstance(i,torch.Tensor) else i if is_listy(i) else list(i)

def hook_input(model, detach=True, grad=False, clone=False):
    return Hook(model, _hook_input, detach=detach, is_forward=not grad, clone=clone)

def is_listy(obj):
    return isinstance(obj, (list, tuple))
