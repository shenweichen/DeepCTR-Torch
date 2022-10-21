# -*- coding:utf-8 -*-
"""
Author:
    zanshuxun, zanshuxun@aliyun.com

Reference:
    [1] Ma X, Zhao L, Huang G, et al. Entire space multi-task model: An effective approach for estimating post-click conversion rate[C]//The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2018.(https://dl.acm.org/doi/10.1145/3209978.3210104)
"""
import torch
import torch.nn as nn

from ..basemodel import BaseModel
from ...inputs import combined_dnn_input
from ...layers import DNN


class ESMM(BaseModel):
    """Instantiates the Entire Space Multi-Task Model architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param tower_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param l2_reg_linear: float, L2 regularizer strength applied to linear part.
    :param l2_reg_embedding: float, L2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: float, L2 regularizer strength applied to DNN.
    :param init_std: float, to use as the initialize std of embedding vector.
    :param seed: integer, to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN.
    :param dnn_use_bn: bool, Whether use BatchNormalization before activation or not in DNN.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss. e.g. ['binary', 'regression'].
    :param task_names: list of str, indicating the predict target of each tasks.
    :param device: str, ``"cpu"`` or ``"cuda:0"``.
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.

    :return: A PyTorch model instance.
    """

    def __init__(self, dnn_feature_columns, tower_dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task_types=('binary', 'binary'),
                 task_names=('ctr', 'ctcvr'), device='cpu', gpus=None):
        super(ESMM, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std,
                                   seed=seed, task='binary', device=device, gpus=gpus)
        self.num_tasks = len(task_names)
        if self.num_tasks != 2:
            raise ValueError("the length of task_names must be equal to 2")
        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")
        if len(task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in task_types:
            if task_type != 'binary':
                raise ValueError("task must be binary in ESMM, {} is illegal".format(task_type))

        input_dim = self.compute_input_dim(dnn_feature_columns)

        self.ctr_dnn = DNN(input_dim, tower_dnn_hidden_units, activation=dnn_activation,
                           dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
        self.cvr_dnn = DNN(input_dim, tower_dnn_hidden_units, activation=dnn_activation,
                           dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)

        self.ctr_dnn_final_layer = nn.Linear(tower_dnn_hidden_units[-1], 1, bias=False)
        self.cvr_dnn_final_layer = nn.Linear(tower_dnn_hidden_units[-1], 1, bias=False)

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.ctr_dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.cvr_dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.ctr_dnn_final_layer.weight, l2=l2_reg_dnn)
        self.add_regularization_weight(self.cvr_dnn_final_layer.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        ctr_output = self.ctr_dnn(dnn_input)
        cvr_output = self.cvr_dnn(dnn_input)

        ctr_logit = self.ctr_dnn_final_layer(ctr_output)
        cvr_logit = self.cvr_dnn_final_layer(cvr_output)

        ctr_pred = self.out(ctr_logit)
        cvr_pred = self.out(cvr_logit)

        ctcvr_pred = ctr_pred * cvr_pred  # CTCVR = CTR * CVR

        task_outs = torch.cat([ctr_pred, ctcvr_pred], -1)
        return task_outs
