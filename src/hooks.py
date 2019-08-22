import torch

class Hook():
    "Create a hook on `m` with `hook_func`."
    def __init__(self, m, hook_func, is_forward=True, detach=True):
        self.hook_func,self.detach,self.stored = hook_func,detach,None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module, input, output):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input  = (o.detach() for o in input ) if is_listy(input ) else input.detach()
            output = (o.detach() for o in output) if is_listy(output) else output.detach()
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed=True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

def _hook_inner(m,i,o):
    return o if isinstance(o,torch.Tensor) else o if is_listy(o) else list(o)

def hook_output(model, detach=True, grad=False):
    return Hook(model, _hook_inner, detach=detach, is_forward=not grad)

def is_listy(obj):
    return (type(obj) == type([])) or (type(obj) == tuple)
