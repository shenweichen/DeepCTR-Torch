from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import DNN, concat_fun


class PNN(BaseModel):

    def __init__(self, linear_feature_columns,
                 dnn_feature_columns, embedding_size=8, dnn_hidden_units=(128, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0, dnn_activation=F.relu, dnn_use_bn=False,
                 use_inner=True, use_outter=False,
                 kernel_type='mat', device='cpu', task='binary'):
        """Instantiates the DeepFM Network architecture.
        :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
        :param embedding_size: positive integer,sparse feature embedding_size
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

        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns, embedding_size, ), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                       init_std=init_std, device=device)

        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.add_regularization_loss(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2_reg_dnn)
        self.add_regularization_loss(self.dnn_linear.weight, l2_reg_dnn)
        self.to(device)

        self.kernel_type = kernel_type
        self.use_inner = use_inner
        self.use_outter = use_outter

        self.embedding_size = embedding_size

        deep_layers = [208, 32, 32]
        field_size = 5
        self.first_order_weight = nn.ModuleList([nn.ParameterList(
            [torch.nn.Parameter(torch.randn(embedding_size), requires_grad=True) for j in range(field_size)])
                                                 for i in range(deep_layers[0])])

        print("Init second order part")
        if self.use_inner:
            self.inner_second_weight_emb = nn.ModuleList([nn.ParameterList(
                [torch.nn.Parameter(torch.randn(self.embedding_size), requires_grad=True) for j in
                 range(field_size)]) for i in range(deep_layers[0])])

        if self.use_outter:
            arr = []
            for i in range(deep_layers[0]):
                tmp = torch.randn(embedding_size, embedding_size)
                arr.append(torch.nn.Parameter(torch.mm(tmp, tmp.t())))
            self.outer_second_weight_emb = nn.ParameterList(arr)
        print("Init second order part finished")

    def forward(self, X):

        if self.kernel_type not in ['mat', 'vec', 'num']:
            raise ValueError("kernel_type must be mat,vec or num")
        """
        # sparse_embedding_list: dnn_feature * batch_size * embedding_size   emb_arr
        # dense_value_list: linear_feature * batch_size * 1
        # linear_signal: batch_size * (dnn_feature * embedding_size)
        # second_product: batch_size * (dnn_feature * embedding_size)
        """
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        cur_bs = len(sparse_embedding_list[0])
        print("sparse_embedding_list: ", len(sparse_embedding_list), len(sparse_embedding_list[0]))
        print("dense_value_list: ", len(dense_value_list), len(dense_value_list[0]))

        linear_signal = concat_fun(sparse_embedding_list).reshape([X.shape[0],
                                                                   len(sparse_embedding_list) * self.embedding_size])
        print("linear_signal", len(linear_signal), len(linear_signal[0]))
        #print(linear_signal[0])

        if self.use_inner:
            inner_product_arr = []
            for i, weight_arr in enumerate(self.inner_second_weight_emb):
                tmp_arr = []
                for j, weight in enumerate(weight_arr):
                    tmp_arr.append(torch.sum(sparse_embedding_list[j].view(cur_bs, self.embedding_size) * weight, 1))
                sum_ = sum(tmp_arr)
                inner_product_arr.append((sum_*sum_).view([-1, 1]))
            inner_product = torch.cat(inner_product_arr, 1)
            second_product = inner_product

        if self.use_outter:
            outer_product_arr = []
            emb_arr_sum = sum(sparse_embedding_list)
            #print("emb_arr_sum", emb_arr_sum)
            emb_matrix_arr = torch.bmm(emb_arr_sum.view([-1, self.embedding_size, 1]),
                                       emb_arr_sum.view([-1, 1, self.embedding_size]))
            print("emb_matrix_arr", len(emb_matrix_arr), len(emb_matrix_arr[0]), emb_matrix_arr.shape)
            for i, weight in enumerate(self.outer_second_weight_emb):
                outer_product_arr.append(torch.sum(torch.sum(emb_matrix_arr*weight, 2), 1).view([-1, 1]))
            outer_product = torch.cat(outer_product_arr, 1)
            second_product = outer_product

        print("out second_product: ", len(second_product), len(second_product[0]))
        # print(second_product[0])

        com = second_product + linear_signal
        # com = torch.cat([linear_signal, second_product], dim=-1).squeeze()
        print("com", len(com), len(com[0]), com.shape)
        # print(com[0])

        dnn_input = combined_dnn_input([com], dense_value_list)
        #dnn_input = combined_dnn_input([second_product], [linear_signal])
        print("dnn_input: ", dnn_input.shape)

        dnn_output = self.dnn(dnn_input)
        # print("dnn_output: ", dnn_output)

        dnn_logit = self.dnn_linear(dnn_output)
        # print("dnn_logit: ", dnn_logit)

        logit = dnn_logit
        # print("logit: ", logit)

        y_pred = self.out(logit)
        # print("y_pred: ", y_pred)

        return y_pred
