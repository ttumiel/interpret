nn.Module.__getitem__ = lambda this,i: list(this.children())[i]
