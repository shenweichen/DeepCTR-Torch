# -*- coding:utf-8 -*-
"""

Author:
    Yuef Zhang

"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Dice(nn.Module):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.

    Input shape:
        - 2 dims: [batch_size, embedding_size(features)]
        - 3 dims: [batch_size, num_features, embedding_size(features)]

    Output shape:
        - Same shape as the input.
    
    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
        - https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
    """
    def __init__(self, num_features, dim=2, epsilon=1e-9):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(num_features, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        if self.dim == 2:
            self.alpha = torch.zeros((num_features,)).to(device)
        else:
            self.alpha = torch.zeros((num_features, 1)).to(device)

    def forward(self, x):
        # x shape: [batch_size, num_features, embedding_size(features)]
        assert x.dim() == 2 or x.dim() == 3

        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        
        return out


def activation_layer(act_name, hidden_size=None, dice_dim=2):
    """Construct activation layers

    Args:
        act_name: str, name of activation function
        hidden_size: int, used for Dice activation
        dice_dim: int, used for Dice activation
    Return:
        act_layer: activation layer
    """
    if act_name.lower() == 'relu' or 'linear':
        act_layer = nn.ReLU(inplace=True)
    elif act_name.lower() == 'dice':
        assert dice_dim
        act_layer = Dice(hidden_size, dice_dim)
    elif act_name.lower() == 'prelu':
        act_layer = nn.PReLU()
    else:
        raise NotImplementedError

    return act_layer


if __name__ == "__main__":
    torch.manual_seed(7)
    a = Dice(3)
    b = torch.rand((5, 3))
    c = a(b)
    print(c.size())
    print('b:', b)
    print('c:', c)
