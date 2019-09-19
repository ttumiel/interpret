import torch
from torch import nn
from scipy.signal import medfilt
import numpy as np
import matplotlib.pyplot as plt

from ..plots import plot

def accuracy(y_hat, y):
    return (y_hat.argmax(-1) == y).float().mean().item()

def init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.)

def conv(ni, nf, ks=3, strd=1, bn=False):
    layers = [nn.Conv2d(ni, nf, ks, padding=ks//2, stride=strd),
              nn.ReLU(True)]
    if bn: layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)

class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

class Learner():
    "A class holding a network that can train and predict on a dataset."
    def __init__(self, data, model, optim=torch.optim.Adam, loss_fn=nn.CrossEntropyLoss, wd=1e-5):
        if isinstance(data, torch.utils.data.DataLoader):
            self.data = data
        else:
            try:
                self.data = torch.utils.data.DataLoader(data, batch_size=64)
            except:
                raise Exception("`data` should be a DataLoader or Dataset.")

        self.model = model
        self.optim_fn = optim
        self.loss_fn = loss_fn
        self.wd = wd
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.losses = []
        self.accs = []

    def fit(self, epochs, lr):
        self.model.train()
        optim = self.optim_fn(self.model.parameters(), lr=lr, weight_decay=self.wd)
        crit = self.loss_fn()

        for epoch in range(epochs):
            for x,y in self.data:
                x,y = x.to(self.device),y.to(self.device)

                preds = self.model(x)
                loss = crit(preds, y)
                self.accs.append(accuracy(preds, y))
                self.losses.append(loss.item())

                loss.backward()
                optim.step()
                optim.zero_grad()

            print(f"{epoch} - loss: {round(np.mean(self.losses[-len(self.data):]),3)}  {accuracy.__name__}: {round(np.mean(self.accs[-len(self.data):]), 3)}")

    def predict(self, data=None, test=True):
        if test: self.model.eval()
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

    def save(self, fn):
        torch.save(self.model.state_dict(), fn)

    def load(self, fn, strict=True):
        self.model.load_state_dict(torch.load(fn), strict=strict)

    def __repr__(self):
        return self.model.__repr__()
