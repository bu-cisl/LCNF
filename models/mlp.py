import torch
import torch.nn as nn
import numpy as np

from models import register

class MLP_Layer(nn.Module):
    def __init__(
        self, in_dim, out_dim, coord_dim=1):
        super(MLP_Layer, self).__init__()
        layer = []
        layer.append(nn.Linear(in_dim+coord_dim, out_dim))
        self.layer = nn.Sequential(*layer)
    def forward(self, x):
        x = self.layer(x)
        return x

class MLP_Layer2(nn.Module):
    def __init__(
        self, in_dim, out_dim, coord_dim=1):
        super(MLP_Layer2, self).__init__()
        layer = []
        layer.append(nn.Linear(in_dim, out_dim))
        self.layer = nn.Sequential(*layer)
    def forward(self, x):
        x = self.layer(x)
        return x

@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            # layers.append(nn.LeakyReLU(0.2))
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]     # input vectors of each point
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)