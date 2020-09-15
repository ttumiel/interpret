"Basic attribution that uses the gradient as the attribution map"

from .attribute import Attribute

class Gradient(Attribute):
    """Uses the gradient of the network with respect to a target class to
    create an attribution map.
    """
    def __init__(self, model, input_img, target_class):
        assert input_img.requires_grad, "Input image must require_grad"
        assert input_img.size(0) == 1, "Input image must have batch size of 1"

        self.m = model.eval()
        self.target_class = target_class
        self.input_data = input_img
        self.calculate_gradient()

    def calculate_gradient(self):
        if self.input_data.grad is not None:
            self.input_data.grad.fill_(0)

        loss = self.m(self.input_data)[0, self.target_class]
        loss.backward()

        self.data = self.input_data.grad.detach().clone().squeeze()
