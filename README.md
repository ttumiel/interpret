# Interpretable Deep Learning

![Class Visualisations](./class_vis.png)

A simple to use PyTorch library for interpreting your deep learning results. Inspired by [TensorFlow Lucid](https://github.com/tensorflow/lucid).

[![Build Status](https://travis-ci.org/ttumiel/interpret.svg?branch=master)](https://travis-ci.org/ttumiel/interpret)
[![Coverage Status](https://coveralls.io/repos/github/ttumiel/interpret/badge.svg?branch=master)](https://coveralls.io/github/ttumiel/interpret?branch=master)

**Note: Repo under construction**

## Installation

Install from PyPI:

```bash
pip install interpret-pytorch
```

Or, install the latest code from GitHub:

```bash
pip install git+https://github.com/ttumiel/interpret
```

### Dependencies

`interpret` requires a working installation of PyTorch.

## Tutorials

Run the tutorials in the browser using Google Colab.

Tutorial | Link
---      | ---
Introduction to `interpret` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ttumiel/interpret/blob/master/nbs/Interpret-Intro.ipynb)

## Usage

`interpret` can be used for both visualisation and attribution. Here an example using a pretrained network is shown.

### Visualisation

```python
from interpret import OptVis, ImageParam, denorm
import torchvision

# Get the PyTorch neural network
network = torchvision.models.vgg11(pretrained=True)

# Select a layer from the network. Use get_layer_names()
# to see a list of layer names and sizes.
layer = 'classifier/6'
neuron = 5

# Create an OptVis object from a PyTorch model
optvis = OptVis.from_layer(network, layer=layer, neuron=neuron)

# Parameterise input noise in colour decorrelated Fourier domain
img_param = ImageParam(224, fft=True, decorrelate=True)

# Create visualisation
optvis.vis(img_param, thresh=(250, 500), transform=True, lr=0.05, wd=0.9)

# Denormalise and return the final image
denorm(img_param())
```

### Attribution

```python
from interpret import gradcam, norm
from PIL import Image
import torchvision

network = torchvision.models.vgg11(pretrained=True)
input_img = Image.open('image.jpg')

# Normalise the input image and turn it into a tensor
input_data = norm(input_img)

# Select the class that we are attributing to
class_number = 207

# Choose a layer for Grad-CAM
layer = 'features/20'

# Generate a Grad-CAM attribution map
saliency_map = gradcam(network, input_data, im_class=class_number, layer=layer)
```
