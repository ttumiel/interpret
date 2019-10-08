import torch
from torch import nn
from scipy.signal import medfilt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from collections import OrderedDict
import time
from IPython.display import display, clear_output

from ..plots import plot, show_images
from ..core import freeze, unfreeze
from ..datasets import DataType
from .callback import OneCycleSchedule

def accuracy(y_hat, y):
    return (y_hat.argmax(-1) == y).float().mean().item()

def init(m):
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    else:
        init_default(m)

def init_default(m, func=nn.init.kaiming_normal_):
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func:
        if hasattr(m, 'weight'): func(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)
    return m

def conv(ni, nf, ks=3, strd=1, bn=False):
    layers = [nn.Conv2d(ni, nf, ks, padding=ks//2, stride=strd),
              nn.ReLU(True)]
    if bn: layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)

def create_head(ni, nf, pool=True):
    if pool:
        layers = [
            AdaptiveConcatPool2d(),
            Flatten()
        ]
        ni = 2*ni
    else:
        layers = []

    layers += [
        nn.BatchNorm1d(ni),
        nn.Dropout(p=0.25),
        nn.Linear(ni, 512),
        nn.ReLU(True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.5),
        nn.Linear(512, nf)
    ]
    return nn.Sequential(*layers).apply(init)

def cut_arch(m, cut=-1):
    body = OrderedDict(list(m.named_children())[:cut])
    return nn.Sequential(body)

class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Learner():
    "A class holding a network that can train and predict on a dataset."
    def __init__(self, data, model, val_data=None, metrics=[], optim=torch.optim.Adam, loss_fn=nn.CrossEntropyLoss, wd=1e-5):
        if isinstance(data, torch.utils.data.DataLoader):
            self.data = data
        else:
            try:
                self.data = torch.utils.data.DataLoader(data, batch_size=64)
            except:
                raise Exception("`data` should be a DataLoader or Dataset.")

        self.val_data = val_data
        self.model = model
        self.optim_fn = optim
        self.loss_fn = loss_fn
        self.wd = wd
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.losses = []
        self.val_losses = []
        self.metrics = metrics
        self.metric_values = [[] for _ in range(len(self.metrics))]

    def fit(self, epochs, lr, callbacks=[]):
        self.model.train()
        self.model.to(self.device)

        # Differential learning rates
        if isinstance(lr, float):
            self.optim = self.optim_fn(self.model.parameters(), lr=lr, weight_decay=self.wd)
        elif isinstance(lr, (list,tuple)):
            assert len(lr) == len(self.model), "lr should have same len() as model"
            p = [{'params': m.parameters(), 'lr': l} for m,l in zip(self.model, lr)]
            self.optim = self.optim_fn(p, weight_decay=self.wd)
        else:
            raise(TypeError("lr should be one of {list, tuple, float}"))

        crit = self.loss_fn()
        table_names = ['train loss']
        if self.val_data is not None:
            table_names += ['val loss'] + [m.__name__ for m in self.metrics]
        table = pd.DataFrame(columns=table_names + ['time'])

        # On train begin
        for cb in callbacks:
            cb.on_train_begin()

        for epoch in range(epochs):
            self.model.train()

            # On epoch begin
            for cb in callbacks:
                cb.on_epoch_begin()

            start_time = time.time()
            pbar = tqdm(self.data, leave=False)
            for x,y in pbar:
                # On batch begin
                for cb in callbacks:
                    cb.on_batch_begin()

                x,y = x.to(self.device),y.to(self.device)

                # Forward Pass
                preds = self.model(x)
                loss = crit(preds, y)
                self.losses.append(loss.item())

                # Backward pass
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

                pbar.set_description_str(f"{round(self.losses[-1], 2)} ")

                # On batch end
                for cb in callbacks:
                    cb.on_batch_end()

            # Validation
            if self.val_data is not None:
                # with torch.no_grad():??
                self.model.eval()
                pbar = tqdm(self.val_data, leave=False)
                val_loss_tmp = []
                metric_tmps = [[] for _ in range(len(self.metrics))]
                for x,y in pbar:
                    x,y = x.to(self.device),y.to(self.device)

                    preds = self.model(x)
                    loss = crit(preds, y)

                    val_loss_tmp.append(loss.item())

                    # Run metrics
                    for i,m in enumerate(self.metrics):
                        metric_tmps[i].append(m(preds, y))

                self.val_losses.append(round(np.mean(val_loss_tmp), 5))
                for i,m in enumerate(self.metrics):
                    self.metric_values[i].append(round(np.mean(metric_tmps[i]), 5))

            end_time = time.time()
            t = end_time-start_time
            trn_loss = round(np.mean(self.losses[-len(self.data):]),5)
            table_data = [trn_loss]

            if self.val_data is not None:
                val_loss = self.val_losses[-1]
                table_data += [val_loss]
                for metr_val in self.metric_values:
                    table_data.append(metr_val[-1])

            time_msg = "{:2}:{:02}".format(round(t//60), round(t%60))
            table.loc[epoch] = table_data + [time_msg]

            clear_output()
            display(table)

            # On epoch end
            for cb in callbacks:
                cb.on_epoch_end()

        # On train end
        for cb in callbacks:
            cb.on_train_end()
        all_preds = []
        if data is None:
            for x,y in self.data:
                x,y = x.to(self.device),y.to(self.device)
                preds = self.model(x)
                all_preds.append(preds)
            return torch.cat(all_preds)
        else:
            if isinstance(data, torch.utils.data.DataLoader):
                for x,y in data:
                    x,y = x.to(self.device),y.to(self.device)
                    preds = self.model(x)
                    all_preds.append(preds)
            else:
                x,y = data
                x,y = x.to(self.device),y.to(self.device)
                preds = self.model(x)
                all_preds.append(preds)
            return torch.cat(all_preds)

    def plot(self, figsize=(10,4), smooth=True):
        f,ax = plt.subplots(1,1+len(self.metrics), figsize=figsize)
        losses = self.losses
        val_mean = np.array(self.val_losses).mean()
        v_losses = [v if v < 2*val_mean else val_mean for v in self.val_losses]
        plot(losses, title="Loss", ax=ax[0], x_lb='Batches', label='train')
        if self.val_data is not None:
            val_x_range = [x*len(self.data) for x in range(1, len(self.val_losses)+1)]
            plot(v_losses, x=val_x_range, title="Loss", ax=ax[0], x_lb='Batches', label='valid')
            for i,m in enumerate(self.metrics):
                plot(self.metric_values[i], x=val_x_range, title=f"Validation {m.__name__}", ax=ax[1], x_lb='Batches')

    def save(self, fn):
        torch.save(self.model.state_dict(), fn)

    def load(self, fn, strict=True):
        self.model.load_state_dict(torch.load(fn), strict=strict)

    def __repr__(self):
        return self.model.__repr__()

    def top_losses(self, n=9, force=False):
        # Reuse existing calculations if they exist
        if hasattr(self, '_top_losses_saved') and not force:
            return self._top_losses_saved

        self.model.eval()
        self.model.to(self.device)
        all_preds = []
        all_ys = []
        all_losses = []

        crit = self.loss_fn()
        single_item_dl = torch.utils.data.DataLoader(self.val_data.dataset, batch_size=1)
        for x,y in tqdm(single_item_dl, leave=False):
            x,y = x.to(self.device),y.to(self.device)
            preds = self.model(x)
            all_preds.append(preds.detach().cpu())
            all_ys.append(y.detach().cpu())
            all_losses.append(crit(preds, y).detach().cpu())

        ps,ys,ls = torch.cat(all_preds), torch.cat(all_ys), torch.stack(all_losses)
        del single_item_dl, all_preds, all_ys, all_losses

        idxs = torch.argsort(ls, descending=True)
        ps = ps.argmax(1)[idxs]
        ys = ys[idxs]
        ls = ls[idxs]
        self._top_losses_saved = ps, ys, ls, idxs
        return ps, ys, ls, idxs

    def plot_top_losses(self, n=9, figsize=(10,10)):
        ps, ys, ls, idxs = self.top_losses()

        ims = torch.stack([self.val_data.dataset[i][0] for i in idxs[:n]])
        labels = [f"{ps[i]}/{ys[i]}" for i in range(n)]

        show_images(ims, normalize=True, figsize=figsize, labels=labels, title="Top Losses: predicted/actual")

    def confusion_matrix(self):
        from sklearn.metrics import confusion_matrix as get_cm

        # TODO: Use the predict method instead of repeating code!
        self.model.to(self.device)
        self.model.eval()
        c = self.data.dataset.c
        cm = np.zeros((c, c))
        for x,y in tqdm(self.val_data, leave=False):
            x,y = x.to(self.device),y.to(self.device)
            preds = self.model(x)
            cm += get_cm(y.detach().cpu().numpy(),
                                preds.argmax(1).detach().cpu().numpy(),
                                labels=np.arange(c))
        return cm

    def plot_confusion_matrix(self):
        cm = self.confusion_matrix().astype('int')

        c = self.data.dataset.c
        ticks = np.arange(c)
        classes = [self.data.dataset.decode_label(i) for i in ticks]
        title = "Confusion Matrix"

        fig, ax = plt.subplots(figsize=(7,6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    #     ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=ticks,
            yticks=ticks,
            xticklabels=classes,
            yticklabels=classes,
            xlim=(-0.5,c-0.5),
            ylim=(c-0.5,-0.5),
            ylabel='True Label',
            xlabel='Predicted Label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        fmt= 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        return ax

    def freeze(self, bn=False):
        freeze(self.model, bn=bn)

    def unfreeze(self):
        unfreeze(self.model)

    def fit_one_cycle(self, epochs, lr, callbacks=[]):
        cb = [OneCycleSchedule(self, lr)] + callbacks
        self.fit(epochs, lr, callbacks=cb)
