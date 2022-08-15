# -*- coding:utf-8 -*-
"""
Author:
    zanshuxun, zanshuxun@aliyun.com

Reference:
    [1] Ruder S. An overview of multi-task learning in deep neural networks[J]. arXiv preprint arXiv:1706.05098, 2017.(https://arxiv.org/pdf/1706.05098.pdf)
"""
import torch
import torch.nn as nn

from ..basemodel import BaseModel
from ...inputs import combined_dnn_input
from ...layers import DNN, PredictionLayer


class SharedBottom(BaseModel):
    """Instantiates the SharedBottom multi-task learning Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param bottom_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of shared bottom DNN.
    :param tower_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param l2_reg_linear: float, L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float, L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float, L2 regularizer strength applied to DNN
    :param init_std: float, to use as the initialize std of embedding vector
    :param seed: integer, to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool, Whether use BatchNormalization before activation or not in DNN
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.

    :return: A PyTorch model instance.
    """

    def __init__(self, dnn_feature_columns, bottom_dnn_hidden_units=(256, 128), tower_dnn_hidden_units=(64,),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task_types=('binary', 'binary'),
                 task_names=('ctr', 'ctcvr'), device='cpu', gpus=None):
        super(SharedBottom, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                           l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding,
                                           init_std=init_std, seed=seed, device=device, gpus=gpus)
        self.num_tasks = len(task_names)
        if self.num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")
        if len(task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in task_types:
            if task_type not in ['binary', 'regression']:
                raise ValueError("task must be binary or regression, {} is illegal".format(task_type))

        self.task_names = task_names
        self.input_dim = self.compute_input_dim(dnn_feature_columns)
        self.bottom_dnn_hidden_units = bottom_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units

        self.bottom_dnn = DNN(self.input_dim, bottom_dnn_hidden_units, activation=dnn_activation,
                              dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                              init_std=init_std, device=device)
        if len(self.tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList(
                [DNN(bottom_dnn_hidden_units[-1], tower_dnn_hidden_units, activation=dnn_activation,
                     dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                     init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.tower_dnn_final_layer = nn.ModuleList([nn.Linear(
            tower_dnn_hidden_units[-1] if len(self.tower_dnn_hidden_units) > 0 else bottom_dnn_hidden_units[-1], 1,
            bias=False) for _ in range(self.num_tasks)])

        self.out = nn.ModuleList([PredictionLayer(task) for task in task_types])

        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.bottom_dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn_final_layer.named_parameters()),
            l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        shared_bottom_output = self.bottom_dnn(dnn_input)

        # tower dnn (task-specific)
        task_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](shared_bottom_output)
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](shared_bottom_output)
            output = self.out[i](tower_dnn_logit)
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)
        return task_outs
