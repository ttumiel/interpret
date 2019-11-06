import torch


class Attribute():
    """Class defining attribution maps over inputs. Contains
    useful plotting methods and implements mathematical
    operations on the underlying data.
    """
    def __init__(self, data):
        self.data = data

    def __add__(self, other):
        if isinstance(other, Attribute):
            return Attribute(self.data + other.data)
        elif isinstance(other, (int, float)):
            return Attribute(self.data + other)
        else:
            raise ValueError(f"Can't add type {type(other)}")

    def __mul__(self, other):
        if isinstance(other, Attribute):
            # TODO: Attempt some shape checking
            return Attribute(self.data * other.data)
        elif isinstance(other, (int, float)):
            return Attribute(self.data * other)
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
