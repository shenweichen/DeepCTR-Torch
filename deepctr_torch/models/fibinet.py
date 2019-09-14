# -*- coding:utf-8 -*-
"""
Author:
    Wutong Zhang
Reference:
    [1] Huang T, Zhang Z, Zhang J. FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1905.09433, 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import SENETLayer
from ..layers.utils import concat_fun



class FiBiNET(BaseModel):
    """Instantiates the Feature Importance and Bilinear feature Interaction NETwork architecture.
    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param embedding_size: positive integer,sparse feature embedding_size
    :param bilinear_type: str,bilinear function type used in Bilinear Interaction Layer,can be ``'all'`` , ``'each'`` or ``'interaction'``
    :param reduction_ratio: integer in [1,inf), reduction ratio used in SENET Layer
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A PyTorch model instance.
    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, embedding_size=8, bilinear_type='interaction',
                 reduction_ratio=3, dnn_hidden_units=(128, 128), l2_reg_linear=1e-5,
                 l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 task='binary', device='cpu'):
        super(FiBiNET, self).__init__(linear_feature_columns, dnn_feature_columns, embedding_size=embedding_size,
                                      dnn_hidden_units=dnn_hidden_units,
                                      l2_reg_linear=l2_reg_linear,
                                      l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn, init_std=init_std,
                                      seed=seed,
                                      dnn_dropout=dnn_dropout, dnn_activation=dnn_activation,
                                      task=task, device=device)
        filed_size = len(self.embedding_dict)
        self.SE = SENETLayer(filed_size, reduction_ratio, seed, device=device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        SE_input = torch.cat(sparse_embedding_list, dim=1)
        SE_output = self.SE(SE_input)
        return SE_output

