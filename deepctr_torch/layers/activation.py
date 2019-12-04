import torch
import torch.nn as nn
import torch.nn.functional as F

class Dice(nn.Module):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.

      Input shape
         - 2D tensor or  3D tensor with shape:  ``(batch_size, N)`` or ``(batch_size, T, N)``.

      Output shape
        - Same shape as the input.

      Arguments
        - **num_features** : Integer, C from an expected input of size (N, C, L)(N,C,L) or LL from input of size (N, L)(N,L).

        - **dim** : Integer, the axis that should be used to compute data distribution (typically the features axis).

        - **epsilon** : Small float added to variance to avoid dividing by zero.

        - **device**: str, ``"cpu"`` or ``"cuda:0"``

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, num_features, epsilon=1e-9, device='cpu', dim=3):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.sigmoid = nn.Sigmoid()
        if dim == 3:
          self.alpha = nn.Parameter(torch.zeros(num_features, 1))
        else:
          self.alpha = nn.Parameter(torch.zeros(1,))
        self.epsilon = epsilon
        self.to(device)
    
    def forward(self, inputs):
        inputs_normed = self.bn(inputs)
        x_p = self.sigmoid(inputs_normed)
        return self.alpha * (1.0 - x_p) * inputs + x_p * inputs