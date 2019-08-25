from torchvision import transforms
from torch import nn

class Blur():
    def __init__(self, kernel_size, variance=1., mean=1.):
        channels=3
        self.filter = nn.Conv2d(channels,channels,kernel_size, bias=False, groups=channels)
        kernel = torch.ones((channels,1,kernel_size,kernel_size))/(kernel_size*kernel_size)

        self.filter.weight.data = kernel
        self.filter.weight.requires_grad_(False)

    def __call__(self, x):
        if x.dim() == 3:
            x.unsqueeze_(0)
        return self.filter(x).squeeze()

def get_transforms(size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], rotate=10, flip_hor=True, flip_vert=False, perspective=True, color_jitter=True, ):
    tfms = [
        transforms.Resize((size, size)),
        transforms.RandomRotation(rotate)
    ]
    if flip_hor: tfms += [transforms.RandomHorizontalFlip()]
    if flip_vert: tfms += [transforms.RandomVerticalFlip()]
    if perspective: tfms += [transforms.RandomPerspective()]

    brightness, contrast, saturation, hue = 0.3, 0.2, 0.2, 0.2
    if color_jitter: tfms += [transforms.ColorJitter(brightness,contrast,saturation,hue)]
    tfms += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    return transforms.Compose(tfms)

def denorm(img):
    return img.add(1).div(2).mul(255).clamp(0,255).permute(1,2,0).cpu().numpy().astype('uint8')
