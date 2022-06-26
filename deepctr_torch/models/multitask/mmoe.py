# -*- coding:utf-8 -*-
"""
Author:
    zanshuxun, zanshuxun@aliyun.com

Reference:
    [1] [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
"""
import torch
import torch.nn as nn

from .basemodel import BaseModel
from .h.inputs import combined_dnn_input
from .h.layers import DNN, PredictionLayer


class MMOELayer(nn.Module):
    """
    The Multi-gate Mixture-of-Experts layer in MMOE model
      Input shape
        - 2D tensor with shape: ``(batch_size,units)``.

      Output shape
        - A list with **num_tasks** elements, which is a 2D tensor with shape: ``(batch_size, output_dim)`` .

      Arguments
        - **input_dim** : Positive integer, dimensionality of input features.
        - **num_tasks**: integer, the number of tasks, equal to the number of outputs.
        - **num_experts**: integer, the number of experts.
        - **output_dim**: integer, the dimension of each output of MMOELayer.

    References
      - [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
    """

    def __init__(self, input_dim, num_tasks, num_experts, output_dim):
        super(MMOELayer, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.expert_network = nn.Linear(self.input_dim, self.num_experts * self.output_dim, bias=True)
        self.gating_networks = nn.ModuleList(
            [nn.Linear(self.input_dim, self.num_experts, bias=False) for _ in range(self.num_tasks)])

    def forward(self, inputs):
        outputs = []
        expert_out = self.expert_network(inputs)
        expert_out = expert_out.reshape([-1, self.output_dim, self.num_experts])
        for i in range(self.num_tasks):
            gate_out = self.gating_networks[i](inputs)
            gate_out = gate_out.softmax(1).unsqueeze(-1)
            output = torch.bmm(expert_out, gate_out).squeeze()
            outputs.append(output)
        return outputs


class MMOE(BaseModel):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param tasks: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param num_experts: integer, number of experts.
    :param expert_dim: integer, the hidden units of each expert.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of shared-bottom DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param device: str, ``"cpu"`` or ``"cuda:0"``

    :return: A PyTorch model instance.
    """

    def __init__(self, dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
                 l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, task_dnn_units=None, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, device='cpu'):
        super(MMOE, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   l2_reg_embedding=l2_reg_embedding, seed=seed, device=device)
        if num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if len(tasks) != num_tasks:
            raise ValueError("num_tasks must be equal to the length of tasks")
        for task in tasks:
            if task not in ['binary', 'regression']:
                raise ValueError("task must be binary or regression, {} is illegal".format(task))

        self.tasks = tasks
        self.task_dnn_units = task_dnn_units
        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                       init_std=init_std, device=device)
        self.mmoe_layer = MMOELayer(dnn_hidden_units[-1], num_tasks, num_experts, expert_dim)
        if task_dnn_units is not None:
            # the last layer of task_dnn should be expert_dim
            self.task_dnn = nn.ModuleList([DNN(expert_dim, task_dnn_units+(expert_dim,)) for _ in range(num_tasks)])
        self.tower_network = nn.ModuleList([nn.Linear(expert_dim, 1, bias=False) for _ in range(num_tasks)])
        self.out = nn.ModuleList([PredictionLayer(task) for task in self.tasks])
        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        dnn_output = self.dnn(dnn_input)
        mmoe_outs = self.mmoe_layer(dnn_output)
        if self.task_dnn_units is not None:
            mmoe_outs = [self.task_dnn[i](mmoe_out) for i, mmoe_out in enumerate(mmoe_outs)]

        task_outputs = []
        for i, mmoe_out in enumerate(mmoe_outs):
            logit = self.tower_network[i](mmoe_out)
            output = self.out[i](logit)
            task_outputs.append(output)

        task_outputs = torch.cat(task_outputs, -1)
        return task_outputs
