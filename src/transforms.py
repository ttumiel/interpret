from torchvision import transforms

def get_transforms(size, mean, std, rotate=10, flip_hor=True, flip_vert=False, perspective=True, color_jitter=True, ):
    tfms = [
        transforms.Resize((size, size)),
        transforms.RandomRotation(rotate)
    ]
    if flip_hor: tfms += [transforms.RandomHorizontalFlip()]
    if flip_vert tfms += [transforms.RandomVerticalFlip()]
    if perspective: tfms += [transforms.RandomPerspective()]

    brightness, contrast, saturation, hue = 0.3, 0.2, 0.2, 0.2
    if color_jitter: tfms += [transforms.ColorJitter(brightness,contrast,saturation,hue)]
    tfms += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    return transforms.Compose(tfms)
