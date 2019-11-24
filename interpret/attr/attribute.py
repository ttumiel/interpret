import matplotlib.pyplot as plt

from interpret.utils import denorm, norm


class Attribute():
    """Class defining attribution maps over inputs. Contains
    useful plotting methods and implements mathematical
    operations on the underlying data.

    Parameters:
        data (torch.Tensor): the attribution map
        input_data (torch.Tensor): the input to the network
    """
    def __init__(self, data, input_data):
        self.data = data
        self.input_data = input_data

    def __add__(self, other):
        if isinstance(other, Attribute):
            # TODO: Add shape checking and generalise from __mul__
            return Attribute(self.data + other.data, self.input_data)
        elif isinstance(other, (int, float)):
            return Attribute(self.data + other, self.input_data)
        else:
            raise ValueError(f"Can't add type {type(other)}")

    def __mul__(self, other):
        if isinstance(other, Attribute):
            self_data, other_data = self.data, other.data

            # Repeat along missing colour dimensions if necessary
            if self.data.ndim < other.data.ndim:
                self_data = self.data.unsqueeze(0).repeat(other.data.shape[0],1,1)
            elif self.data.ndim > other.data.ndim:
                other_data = other.data.unsqueeze(0).repeat(self.data.shape[0],1,1)

            # Compare shape of data tensors
            if self_data.shape != other_data.shape:
                self_data = denorm(self_data)
                other_data = denorm(other_data)
                if self_data.size > other_data.size:
                    other_data = other_data.resize(self_data.size, resample=2)
                else:
                    self_data = self_data.resize(other_data.size, resample=2)

                self_data = norm(self_data, unsqueeze=False, grad=False)
                other_data = norm(other_data, unsqueeze=False, grad=False)

            return Attribute(self_data * other_data, self.input_data)
        elif isinstance(other, (int, float)):
            return Attribute(self.data * other, self.input_data)
        else:
            raise ValueError(f"Can't multiply by type {type(other)}")

    def __sub__(self, other):
        return self + (-1*other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self.__mul__(-1.)

    # TODO: Generalise this method to non-image data
    def show(self, ax=None, show_image=True, alpha=0.4, cmap='magma', colorbar=False):
        """Show the generated attribution map.

        Parameters:
            show_image (bool): show the denormalised input image overlaid on the heatmap.
            ax: axes on which to plot image.
            colorbar (bool): show a colorbar.
            cmap: matplotlib colourmap.
            alpha (float): transparency value alpha for heatmap.
        """
        if ax is None:
            _,ax = plt.subplots()

        sz = list(self.input_data.shape[2:])
        if show_image:
            input_image = denorm(self.input_data[0])
            ax.imshow(input_image)

        data = self.data
        if (data < 0).any():
            data = (data-data.min())/(data.max()-data.min())

        if data.ndim >= 3:
            data = data.squeeze()
            if data.ndim == 3:
                data = data.permute(1,2,0)
            else:
                raise RuntimeError(f"Can't display data shape {self.data.shape} as an image.")

        im = ax.imshow(data, alpha=alpha, extent=(0,*sz[::-1],0), interpolation='bilinear', cmap=cmap)
        if colorbar:
            ax.figure.colorbar(im, ax=ax)
