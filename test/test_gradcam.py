from interpret import gradcam, norm
from PIL import Image
import torchvision, pytest, torch

def test_gradcam():
    network = torchvision.models.vgg11(pretrained=False)
    input_img = torch.randn(1,3,224,224)

    class_number = 207
    layer = 'features/20'
    saliency_map = gradcam(network, input_img, im_class=class_number, layer=layer)

    assert list(saliency_map.shape) == [7,7]
