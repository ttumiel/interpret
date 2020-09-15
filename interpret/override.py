"Override functions in a nn.Module"

from interpret.utils import find_all
from interpret.models.layers import GeneralizedReLU


class ModuleOverride():
    def __init__(self, model, module, override):
        self.model = model
        self.module = module
        self.override = override

    def __enter__(self):
        self.originals, self.paths = find_all(self.model, self.module, path=True)
        for p in self.paths:
            self.model[p] = self.override
        return self

    def __exit__(self, *args):
        for i,p in enumerate(self.paths):
            self.model[p] = self.originals[i]

class ReLUOverride(ModuleOverride):
    def __init__(self, model, **generalized_relu_kwargs):
        super().__init__(model, nn.ReLU, GeneralizedReLU(**generalized_relu_kwargs))
