import torch
from torch import nn
from scipy.signal import medfilt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict
import time

from ..plots import plot

def accuracy(y_hat, y):
    return (y_hat.argmax(-1) == y).float().mean().item()

def init_head(m):
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
    return nn.Sequential(*layers).apply(init_head)

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
    def __init__(self, data, model, val_data=None, optim=torch.optim.Adam, loss_fn=nn.CrossEntropyLoss, wd=1e-5):
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
        self.accs = []
        self.val_losses = []
        self.val_accs = []

    def fit(self, epochs, lr):
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

        for epoch in range(epochs):
            self.model.train()
            start_time = time.time()
            pbar = tqdm(self.data, leave=False)
            for x,y in pbar:
                x,y = x.to(self.device),y.to(self.device)

                preds = self.model(x)
                loss = crit(preds, y)
                self.accs.append(accuracy(preds, y))
                self.losses.append(loss.item())

                loss.backward()
                optim.step()
                optim.zero_grad()

                pbar.set_description_str(f"{round(self.losses[-1], 2)} ")

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
                    # TODO: val_* should be averages at the end of every epoch
                    self.val_accs.append(accuracy(preds, y))
                    self.val_losses.append(loss.item())

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
        f,ax = plt.subplots(1,2, figsize=figsize)
        losses, accs = self.losses, self.accs
        if smooth: losses, accs = medfilt(self.losses, 5), medfilt(self.accs, 5)
        plot(losses, title="Loss", ax=ax[0], x_lb='Batches')
        plot(accs, title="Accuracy", ax=ax[1], x_lb='Batches')
        if self.val_data is not None:
            # TODO: Add legend
            plot(self.val_losses, title="Loss", ax=ax[0], x_lb='Batches')
            plot(self.val_accs, title="Accuracy", ax=ax[1], x_lb='Batches')


    def save(self, fn):
        torch.save(self.model.state_dict(), fn)

    def load(self, fn, strict=True):
        self.model.load_state_dict(torch.load(fn), strict=strict)

    def __repr__(self):
        return self.model.__repr__()
