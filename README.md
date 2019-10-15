# Interpretable Deep Learning

![Class Visualisations](./class_vis.png)

A simple to use PyTorch library for interpreting your deep learning results.

**Note: Repo under construction**

## Installation

Currently, install from GitHub:

```bash
pip install git+https://github.com/ttumiel/interpret
```

### Dependencies

`interpret` requires a working installation of PyTorch.

## Usage

`interpret` can be used for both visualisation and attribution.

### Visualisation

```python
network = get_network(...)
layer = 'fc'
neuron = 5

# Create an OptVis object from a PyTorch model
optvis = OptVis.from_layer(network, layer=layer, neuron=neuron)

# Parameterise input noise
img_param = ImageParam(224, fft=True, decorrelate=True)

# Create visualisation
optvis.vis(img_param, thresh=(250, 500), transform=True, lr=0.05, wd=0.9)

# View the output visualisation
denorm(img_param())
```

### Attribution

```python
network = get_network(...)
input_img = get_image(...)
class_number = 5
layer = ... choose a layer

# Generate a Grad-CAM attribution map
saliency_map = gradcam(network, input_img, im_class=class_number, layer=layer)
```
