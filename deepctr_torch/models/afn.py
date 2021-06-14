# -*- coding:utf-8 -*-
"""
Author:
    Weiyu Cheng, weiyu_cheng@sjtu.edu.cn

Reference:
    [1] Cheng, W., Shen, Y. and Huang, L. 2020. Adaptive Factorization Network: Learning Adaptive-Order Feature
         Interactions. Proceedings of the AAAI Conference on Artificial Intelligence. 34, 04 (Apr. 2020), 3609-3616.
"""
import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..layers import LogTransformLayer, DNN


class AFN(BaseModel):
    """Instantiates the Adaptive Factorization Network architecture.
    
    In DeepCTR-Torch, we only provide the non-ensembled version of AFN for the consistency of model interfaces. For the ensembled version of AFN+, please refer to https://github.com/WeiyuCheng/DeepCTR-Torch (Pytorch Version) or https://github.com/WeiyuCheng/AFN-AAAI-20 (Tensorflow Version).

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param ltl_hidden_size: integer, the number of logarithmic neurons in AFN
    :param afn_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of DNN layers in AFN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.
    
    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns,
                 ltl_hidden_size=256, afn_dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 task='binary', device='cpu', gpus=None):

        super(AFN, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device, gpus=gpus)

        self.ltl = LogTransformLayer(len(self.embedding_dict), self.embedding_size, ltl_hidden_size)
        self.afn_dnn = DNN(self.embedding_size * ltl_hidden_size, afn_dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=True,
                       init_std=init_std, device=device)
        self.afn_dnn_linear = nn.Linear(afn_dnn_hidden_units[-1], 1)
        self.to(device)
    
    def forward(self, X):

        sparse_embedding_list, _ = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                   self.embedding_dict)
        logit = self.linear_model(X)
        if len(sparse_embedding_list) == 0:
            raise ValueError('Sparse embeddings not provided. AFN only accepts sparse embeddings as input.')
            
        afn_input = torch.cat(sparse_embedding_list, dim=1)
        ltl_result = self.ltl(afn_input)
        afn_logit = self.afn_dnn(ltl_result)
        afn_logit = self.afn_dnn_linear(afn_logit)
        
        logit += afn_logit
        y_pred = self.out(logit)
        
        return y_pred
