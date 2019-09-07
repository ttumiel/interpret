from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from utils import *

class ImageBunch(Dataset):
    def __init__(self, train_dl, valid_dl, test_dl=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl

    @classmethod
    def from_folder(cls, path, bs=64, shuffle=True, tfms=None):
        return cls(ImageFolder(Path(path)/'train', transform=tfms), ImageFolder(Path(path)/'valid', transform=tfms))

def random_shapes(size, shape, min_size, max_size, coord_limits, background, color=True):
    "Creates images of randomly sized shapes. One shape of 'triangle', 'rectangle' or 'circle'."
    # make sure that the shape will be visible within the given coord limits
    if background=="uniform":
        arr = np.ones((size, size, 3 if color else 1), dtype="uint8")
    else:
        arr = (np.random.rand(size, size, 3 if color else 1)*255).astype("uint8")

    im = Image.fromarray(arr)
    draw = ImageDraw.Draw(im)

    def draw_circle():
        x,y = np.random.randint(*coord_limits[:2]), np.random.randint(*coord_limits[2:])
        length = np.random.randint(min_size, max_size)
        coords = [x,y,x+length,y+length]
        fill = tuple(np.random.randint(0,256,3))
        draw.ellipse(coords, fill=fill)

    def draw_rect():
        x,y = np.random.randint(*coord_limits[:2]), np.random.randint(*coord_limits[2:])
        l1 = np.random.randint(min_size, max_size)
        l2 = np.random.randint(min_size, max_size)
        coords = [x,y,x+l1,y+l2]
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
    fn()
    return im, label
