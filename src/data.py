from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from .utils import *
import numpy as np
from PIL import Image, ImageDraw
import random

class ImageBunch(Dataset):
    def __init__(self, train_dl, valid_dl, test_dl=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl

    @classmethod
    def from_folder(cls, path, bs=64, shuffle=True, tfms=None):
        return cls(ImageFolder(Path(path)/'train', transform=tfms), ImageFolder(Path(path)/'valid', transform=tfms))

def random_shapes(size, shape, min_size, max_size, coord_limits, background, color=True, number=1):
    """
    coord_limits: [x_min, x_max, y_min, y_max]. If None, defaults to edges of size.
    """
    if coord_limits is None:
        coord_limits = [0, size, 0, size]

    bgs = ["uniform", "random", "color"]
    assert background in bgs, f'Choose background from {bgs}'
    if background=="uniform":
        arr = np.ones((size, size, 3 if color else 1), dtype="uint8")
    elif background=="random":
        arr = (np.random.rand(size, size, 3 if color else 1)*255).astype("uint8")
    elif background=="color":
        arr = (np.ones((size, size, 3 if color else 1))*np.random.rand(1,1,3)*255).astype("uint8")

    im = Image.fromarray(arr)
    draw = ImageDraw.Draw(im)

    def draw_circle():
        x0,y0 = np.random.randint(*coord_limits[:2]), np.random.randint(*coord_limits[2:])
        l = np.random.randint(min_size, max_size)
        x1 = x0+l
        y1 = y0+l
        if x0>x1: x0,x1=x1,x0
        if y0>y1: y0,y1=y1,y0
        coords = [x0,y0,x1,y1]
        fill = tuple(np.random.randint(0,256,3))
        draw.ellipse(coords, fill=fill)

    def draw_rect():
        x0,y0 = np.random.randint(*coord_limits[:2]), np.random.randint(*coord_limits[2:])
        l1 = np.random.randint(min_size, max_size)
        l2 = np.random.randint(min_size, max_size)
        x1 = x0+l1 if x0>coord_limits[1]/2 else x0-l1
        y1 = y0+l2 if y0>coord_limits[3]/2 else x0-l2
        coords = [x0, y0 ,x1, y1]
        fill = tuple(np.random.randint(0,256,3))
        draw.rectangle(coords, fill=fill)

    def draw_tri():
        coords = [(np.random.randint(*coord_limits[:2]), np.random.randint(*coord_limits[2:])) for _ in range(3)]
        fill = tuple(np.random.randint(0,256,3))
        draw.polygon(coords,fill=fill)

    SHAPES = {'rectangle': (draw_rect, 0), 'circle': (draw_circle, 1), 'triangle': (draw_tri,2)}
    if shape is None:
        fn, label = random.choice(list(SHAPES.values()))
    else:
        fn, label = SHAPES[shape]
    for _ in range(number):
        fn()

    return im, label
