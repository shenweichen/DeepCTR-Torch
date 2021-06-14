# -*- coding:utf-8 -*-
"""
Author:
    zanshuxun, zanshuxun@aliyun.com
Reference:
    [1] Yu Y, Wang Z, Yuan B. An Input-aware Factorization Machine for Sparse Prediction[C]//IJCAI. 2019: 1466-1472.(https://www.ijcai.org/Proceedings/2019/0203.pdf)
"""
import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input, SparseFeat, VarLenSparseFeat
from ..layers import FM, DNN


class IFM(BaseModel):
    """Instantiates the IFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on ``device`` .  ``gpus[0]``  should be the same gpu with ``device`` .
    :return: A PyTorch model instance.

    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):
        super(IFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device, gpus=gpus)

        if not len(dnn_hidden_units) > 0:
            raise ValueError("dnn_hidden_units is null!")

        self.fm = FM()

        self.factor_estimating_net = DNN(self.compute_input_dim(dnn_feature_columns, include_dense=False),
                                         dnn_hidden_units, activation=dnn_activation, l2_reg=l2_reg_dnn,
                                         dropout_rate=dnn_dropout,
                                         use_bn=dnn_use_bn, init_std=init_std, device=device)
        self.sparse_feat_num = len(list(filter(lambda x: isinstance(x, SparseFeat) or isinstance(x, VarLenSparseFeat),
                                               dnn_feature_columns)))
        self.transform_weight_matrix_P = nn.Linear(
            dnn_hidden_units[-1], self.sparse_feat_num, bias=False).to(device)

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.factor_estimating_net.named_parameters()),
            l2=l2_reg_dnn)
        self.add_regularization_weight(self.transform_weight_matrix_P.weight, l2=l2_reg_dnn)

        self.to(device)

    def forward(self, X):
        sparse_embedding_list, _ = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        if not len(sparse_embedding_list) > 0:
            raise ValueError("there are no sparse features")

        dnn_input = combined_dnn_input(sparse_embedding_list, [])  # (batch_size, feat_num * embedding_size)
        dnn_output = self.factor_estimating_net(dnn_input)
        dnn_output = self.transform_weight_matrix_P(dnn_output)  # m'_{x}
        input_aware_factor = self.sparse_feat_num * dnn_output.softmax(1)  # input_aware_factor m_{x,i}

        logit = self.linear_model(X, sparse_feat_refine_weight=input_aware_factor)

        fm_input = torch.cat(sparse_embedding_list, dim=1)
        refined_fm_input = fm_input * input_aware_factor.unsqueeze(-1)  # \textbf{v}_{x,i}=m_{x,i}\textbf{v}_i
        logit += self.fm(refined_fm_input)

        y_pred = self.out(logit)

        return y_pred
