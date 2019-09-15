from pathlib import Path
from PIL import Image

Path.ls = lambda c: list(c.iterdir())

def zoom(im, zoom=2):
    return im.transform((int(im.size[0]*zoom), int(im.size[1]*zoom)), Image.EXTENT, (0, 0, im.size[0], im.size[1]))