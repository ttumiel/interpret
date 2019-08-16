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
