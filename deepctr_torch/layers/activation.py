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
    """
    
    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """
    def __init__(self, num_features, dim=2):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(num_features, eps=1e-9)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        
        if self.dim == 3:
            self.alpha = torch.zeros((num_features, 1)).to(device)
        elif self.dim == 2:
            self.alpha = torch.zeros((num_features,)).to(device)
        

    def forward(self, x):
        if self.dim == 3:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        
        elif self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        
        return out


if __name__ == "__main__":
    a = Dice(32, dim=3)
    # b = torch.zeros((10, 32))
    b = torch.zeros((10, 100, 32))
    c = a(b)
    print(c.size())
