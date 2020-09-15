import torch
import math
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from interpret import show_images, Gradcam, LayerObjective

__all__ = [
    'validate', 'top_losses', 'plot_top_losses',
    'confusion_matrix', 'plot_confusion_matrix',
    'get_dataset_examples', 'plot_dataset_examples'
    ]

def validate(network, dataloader, metrics=None, device=None):
    """Validate a dataset on a trained network with a set of metrics.

    Returns (tuple):
        Tuple of (predictions, labels, *metrics)
    """
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    network.eval().to(device)
    if metrics is None: metrics = []

    all_preds = []
    all_ys = []
    all_metrics = [[]*len(metrics)]

    with torch.no_grad():
        for x,y in tqdm(dataloader, leave=False):
            x,y = x.to(device),y.to(device)
            preds = network(x)
            all_preds.append(preds.detach())
            all_ys.append(y.detach())
            for i,m in enumerate(metrics):
                loss = m(preds, y)
                all_metrics[i].append(loss.detach())

    m = tuple(torch.cat(m) for m in all_metrics if len(m)>0)
    p,y = torch.cat(all_preds), torch.cat(all_ys)
    del all_preds, all_ys, all_metrics
    return (p, y) + m

def top_losses(network, dataloader, loss_fn, device=None):
    """Find the dataset examples that result in the largest loss.

    Parameters:
        network (nn.Module): the trained network on which to search
            for the top losses.
        dataloader (DataLoader): a dataloader containing the inputs
            to the network.
        loss_fn (Callable): the loss function that is minimised.

    Returns: (Tensor, Tensor, Tensor, Tensor)
        A tuple of the sorted predictions, targets, losses,
        and the indexes in the dataset.
    """
    _check_shuffle(dataloader)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    network.eval().to(device)
    if hasattr(loss_fn, 'reduction'):
        loss_fn.reduction = 'none'
    else:
        assert dataloader.batch_size == 1, 'If loss function has no reduction attribute then batch_size must equal 1.'

    p,y,l = validate(network, dataloader, [loss_fn], device)

    idxs = torch.argsort(l, descending=True)
    return p[idxs], y[idxs], l[idxs], idxs

def plot_top_losses(network, dataloader, loss_fn, *top_losses_out, device=None,
                    n=9, figsize=(10,10), gradcam=False, layer=0, show_image=True):
    """Plot the top losses that are returned by `top_losses()`

    Inspired by fastai's plot_top_losses method.

    Parameters:
        network (nn.Module): The trained model.
        dataloader (DataLoader): Validation dataloader. Shuffle=False.
        loss_fn (Callable): The function that generates the loss.
        top_losses_out: the output of top_losses() to prevent rerunning.
        device: torch.device
        n (int): number of inputs to plot.
        figsize (tuple): figure size
        gradcam (bool): plot the target class' gradcam saliency map over
            the images.
        layer: the layer whose output to generate the Gradcam map from.
        show_image (bool): whether to show the image - useful if the gradcam
            is hard to see.

    Returns (Tensor, Tensor, Tensor, Tensor)
        The sorted predictions, targets, losses and all ranked
        indexes of all the inputs.
    """
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    network.eval().to(device)

    if len(top_losses_out) == 0:
        ps, ys, ls, idxs = top_losses(network, dataloader, loss_fn, device)
    else:
        ps, ys, ls, idxs = top_losses_out

    p,y,l = ps[:n], ys[:n], ls[:n]

    if p.ndim > 1:
        p = p.argmax(1)

    ims = torch.stack([dataloader.dataset[i.item()][0] for i in idxs[:n]])
    labels = ["{}/{:d}\n{:.2f}".format(p[i], y[i].to(torch.long), l[i].item()) for i in range(n)]
    title = "Top Losses\npredicted/actual\nloss"

    # Can't use show_images because we have to add heatmap
    if gradcam:
        r = math.ceil(math.sqrt(n))
        f,axes = plt.subplots(r,r,figsize=figsize)
        f.suptitle(title, weight='bold', fontsize=16)
        for i,ax in enumerate(axes.flatten()):
            if i<n:
                heatmap = Gradcam(network, ims[i][None].to(device), y[i], layer=layer)
                heatmap.show(show_image=show_image, ax=ax)
                ax.set_title(labels[i])
            ax.set_axis_off()
    else:
        show_images(ims, normalize=True, figsize=figsize, labels=labels, title=title)

    return ps,ys,ls,idxs


def confusion_matrix(network, dataloader, num_classes, device=None):
    """Create a confusion matrix from network

    Parameters:
        network (nn.Module): The trained network.
        dataloader (DataLoader): the validation dataloader.
        num_classes (int): the number of classes that are being optimised for.
        device: torch.device

    Returns np.ndarray:
        A (num_classes by num_classes) numpy array. True values on
        the left, predicted along the bottom.
    """
    from sklearn.metrics import confusion_matrix as get_cm

    preds,y = validate(network, dataloader, [], device)
    preds = preds.argmax(1) if preds.ndim > 1 else preds.round()
    preds = preds.detach().squeeze().cpu().numpy()
    y = y.detach().squeeze().cpu().numpy()

    return get_cm(y, preds, labels=np.arange(num_classes)).astype(int)

def plot_confusion_matrix(num_classes, cm=None, network=None, dataloader=None,
                          device=None, decode_label=None):
    "Plot the confusion matrix created by `confusion_matrix`"
    assert cm is not None or (network is not None and dataloader is not None)
    if cm is None:
        cm = confusion_matrix(network, dataloader, num_classes).astype('int')
    else:
        assert cm.shape[0] == num_classes and cm.shape[1] == num_classes
        cm = cm.astype(int)

    ticks = np.arange(num_classes)
    classes = [decode_label[i] for i in ticks] if decode_label is not None else ticks

    fig, ax = plt.subplots(figsize=(7,6))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=classes,
        yticklabels=classes,
        xlim=(-0.5,num_classes-0.5),
        ylim=(num_classes-0.5,-0.5),
        ylabel='True Label',
        xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def get_dataset_examples(network, dataloader, layer, channel=None, device=None, **layer_kwargs):
    """Sample dataset examples that most highly activate a particular
    LayerObjective. Useful to decipher some of the more obscure looking
    visualisations.

    Parameters:
        network (nn.Module): The trained network.
        dataloader (DataLoader): the validation dataloader. Shuffle
            must be False.
        layer (str): the layer to optimise using LayerObjective.
        channel (int): the channel to optimise within a layer.
        device: torch.device
        **layer_kwargs: additional keyword arguments to pass to LayerObjective.

    Returns (Tensor):
        The sorted indices of the items that most activate the layer.
    """
    _check_shuffle(dataloader)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    network.eval().to(device)
    obj = LayerObjective(network, layer, channel, batchwise=True, **layer_kwargs)

    losses = []
    with torch.no_grad():
        for x,y in tqdm(dataloader, leave=False):
            x = x.to(device)
            loss = obj(x)
            losses.append(loss.detach())

    losses = torch.cat(losses)
    idxs = losses.argsort()
    return idxs

def plot_dataset_examples(n, dataloader, denorm=True, network=None, layer=None,
                          idxs=None, channel=None, figsize=(10,10), **layer_kwargs):
    "Plot the dataset examples generated by `get_dataset_examples`"
    assert (network is not None and layer is not None) or idxs is not None, \
            'Must supply either network and layer args or idxs from `get_dataset_examples`'
    if idxs is None:
        idxs = get_dataset_examples(network, dataloader, layer, channel, **layer_kwargs)
    title = 'Dataset Examples'
    ims, labels = list(zip(*[dataloader.dataset[i] for i in idxs[:n]]))
    ims = torch.stack(ims)
    show_images(ims, denorm, title=title, labels=labels, figsize=figsize)
    return idxs

def _check_shuffle(dl):
    assert not isinstance(dl.sampler, torch.utils.data.RandomSampler), \
        "DataLoader should not be shuffled."
