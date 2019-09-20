from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import DNN, concat_fun
from ..layers.interaction import InnerProductLayer, OutterProductLayer


class PNN(BaseModel):

    def __init__(self, linear_feature_columns,
                 dnn_feature_columns, embedding_size=8, field_size=5, dnn_hidden_units=(128, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0, dnn_activation=F.relu, dnn_use_bn=False,
                 use_inner=True, use_outter=False,
                 kernel_type='mat', device='cpu', task='binary'):
        """Instantiates the DeepFM Network architecture.
        :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
        :param embedding_size: positive integer,sparse feature embedding_size
        :param field_size: a field contains several embedding feature
        :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
        :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
        :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
        :param init_std: float,to use as the initialize std of embedding vector
        :param seed: integer ,to use as random seed.
        :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
        :param dnn_activation: Activation function to use in DNN
        :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
        :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
        :param device:
        :return: A PyTorch model instance.
        """

        super(PNN, self).__init__(linear_feature_columns, dnn_feature_columns, embedding_size=embedding_size,
                                  dnn_hidden_units=dnn_hidden_units,
                                  l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn,
                                  l2_reg_linear=l2_reg_linear, init_std=init_std, seed=seed,
                                  dnn_dropout=dnn_dropout, dnn_activation=dnn_activation,
                                  task=task, device=device)

        if use_inner and use_outter:
            self.order_len = self.compute_order_dim(dnn_feature_columns, embedding_size)
            order_to_dnn = self.compute_order_dim(dnn_feature_columns, embedding_size)*2

        elif use_inner or use_outter:
            self.order_len = self.compute_order_dim(dnn_feature_columns, embedding_size)
            order_to_dnn = self.order_len
        else:
            self.order_len = 0
            order_to_dnn = 0

        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns, embedding_size, ) + order_to_dnn, dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                       init_std=init_std, device=device)

        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.add_regularization_loss(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2_reg_dnn)
        self.add_regularization_loss(self.dnn_linear.weight, l2_reg_dnn)

        self.kernel_type = kernel_type
        self.use_inner = use_inner
        self.use_outter = use_outter

        self.embedding_size = embedding_size
        self.field_size = field_size
        # Product Component

        if self.use_inner:
            self.innerproduct = InnerProductLayer(self.order_len, self.embedding_size, field_size, device=device)
            """
            print("Init IPNN component")
            self.ipnn_weight_embed = nn.ModuleList([nn.ParameterList(
                [torch.nn.Parameter(torch.randn(self.embedding_size), requires_grad=True) for j in
                 range(self.field_size)]) for i in range(self.order_len)])
            print("Init IPNN component finished")"""

        if self.use_outter:
            self.outterproduct = OutterProductLayer(self.order_len, self.embedding_size, field_size, device=device)
            """
            print("Init OPNN component")
            arr = []
            for i in range(self.order_len):
                tmp = torch.randn(embedding_size, embedding_size)
                arr.append(torch.nn.Parameter(torch.mm(tmp, tmp.t())))
            self.opnn_weight_embed = nn.ParameterList(arr)
            print("Init OPNN component finished")
            """

        self.to(device)

    def forward(self, X):

        if self.kernel_type not in ['mat', 'vec', 'num']:
            raise ValueError("kernel_type must be mat,vec or num")
        """
        # sparse_embedding_list: dnn_feature * batch_size * embedding_size   emb_arr
        # dense_value_list: linear_feature * batch_size * 1
        # linear_signal: batch_size * (dnn_feature * embedding_size)
        # second_product: batch_size * (dnn_feature * embedding_size)
        # product_layer = linear_signal + second_product = 208 * 2 = 416
        # dnn_input = com_product + dense_value_list = 416 + 13 = 419
        """
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        cur_bs = len(sparse_embedding_list[0])
        linear_signal = concat_fun(sparse_embedding_list).reshape([X.shape[0],
                                                                   len(sparse_embedding_list) * self.embedding_size])

        if self.use_inner:
            """
            inner_product_arr = []
            for i, weight_arr in enumerate(self.ipnn_weight_embed):
                tmp_arr = []
                for j, weight in enumerate(weight_arr):
                    tmp_arr.append(torch.sum(sparse_embedding_list[j].view(cur_bs, self.embedding_size) * weight, 1))
                sum_ = sum(tmp_arr)
                inner_product_arr.append((sum_*sum_).view([-1, 1]))
            inner_product = torch.cat(inner_product_arr, 1)
            """
            inner_product = self.innerproduct(sparse_embedding_list, cur_bs)

        if self.use_outter:
            """
            outer_product_arr = []
            emb_arr_sum = sum(sparse_embedding_list)
            emb_matrix_arr = torch.bmm(emb_arr_sum.view([-1, self.embedding_size, 1]),
                                       emb_arr_sum.view([-1, 1, self.embedding_size]))
            for i, weight in enumerate(self.opnn_weight_embed):
                outer_product_arr.append(torch.sum(torch.sum(emb_matrix_arr*weight, 2), 1).view([-1, 1]))
            outer_product = torch.cat(outer_product_arr, 1)
            """
            outer_product = self.outterproduct(sparse_embedding_list, cur_bs)

        if self.use_outter and self.use_inner:
            product_layer = torch.cat([linear_signal, inner_product, outer_product], dim=1)
        elif self.use_outter:
            product_layer = torch.cat([linear_signal, outer_product], dim=1)
        elif self.use_inner:
            product_layer = torch.cat([linear_signal, inner_product], dim=1)
        else:
            product_layer = linear_signal

        dnn_input = combined_dnn_input([product_layer], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        logit = dnn_logit

        y_pred = self.out(logit)

        return y_pred
