# -*- coding:utf-8 -*-
"""
Author:
    hugo.guo

Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import DNN, concat_fun, AttentionSequencePoolingLayer, Dice

class DIN(BaseModel):
    """Instantiates the Deep Interest Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_embedding_size: positive integer, sparse feature embedding_size.
    :param hist_embedding_size: positive integer, history feature embedding_size.
    :param hist_len_max: positive int, to indicate the max length of seq input
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :return: A PyTorch model instance.

    """
    def __init__(self, linear_feature_columns, dnn_feature_columns, history_feature_list, dnn_embedding_size=8, hist_embedding_size=8, hist_len_max=16, 
                 dnn_use_bn=False, dnn_hidden_units=(200, 80), dnn_activation=F.relu, att_hidden_size=(80, 40), att_activation="dice",
                 att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024,
                 task='binary', device='cpu'):
        super(DIN, self).__init__(linear_feature_columns, dnn_feature_columns, embedding_size=dnn_embedding_size,
                        dnn_hidden_units=dnn_hidden_units, l2_reg_embedding=l2_reg_embedding, 
                        l2_reg_dnn=l2_reg_dnn, init_std=init_std, seed=seed, dnn_dropout=dnn_dropout, 
                        dnn_activation=dnn_activation, task=task, device=device)

        self.att_hidden_size = att_hidden_size
        self.att_activation = att_activation

        # history_list
        self.history_feature_list = list(map(lambda x: "hist_" + x, history_feature_list))
        
        # query_list
        self.query_feature_list = history_feature_list

        # out of query_list and history_list
        self.dnn_feature_columns_outof_hist = [var for var in dnn_feature_columns if var.name not in (self.query_feature_list + self.history_feature_list)]

        if len(dnn_feature_columns) == 0 or len(dnn_hidden_units) == 0:
            raise ValueError("dnn feature columns or dnn hidden units must be provided")
        
        # hist_feats
        self.hist_feats = [var for var in dnn_feature_columns if var.name in self.history_feature_list]
        # query_feats
        self.query_feats = [var for var in dnn_feature_columns if var.name in self.query_feature_list]

        # update history feats embedding size
        for key in self.query_feature_list:
            self.embedding_dict[key].embedding_dim = hist_embedding_size

        # att_pool
        keys_len = self.feature_index[self.hist_feats[0].name][1] - self.feature_index[self.hist_feats[0].name][0]
        if isinstance(self.att_activation, str):
            if self.att_activation.lower() == 'dice':
                att_activation_list = nn.ModuleList(Dice(keys_len, device=device, dim=3) for i in range(len(self.att_hidden_size)))
            elif self.att_activation.lower() == 'relu':
                att_activation_list = F.relu
            elif self.att_activation.lower() == 'tanh':
                att_activation_list = F.tanh
            elif self.att_activation.lower() == 'prelu':
                att_activation_list = F.prelu           
        else:
            att_activation_list = self.att_activation

        self.att_pool = AttentionSequencePoolingLayer(len(self.query_feature_list) * hist_embedding_size, att_hidden_size, att_activation_list,
                                         weight_normalization=att_weight_normalization, init_std=init_std,
                                         use_bn=False, dropout_rate=0, l2_reg=0, seed=seed, device=device)
        
        self.add_regularization_loss(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.att_pool.local_att.dnn.named_parameters()), l2_reg_dnn)
        self.add_regularization_loss(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.att_pool.local_att.dnn_linear.named_parameters()), l2_reg_dnn)
        
        # DNN
        self.dnn = DNN(self.compute_input_dim(self.dnn_feature_columns_outof_hist, dnn_embedding_size) + len(self.query_feats) * hist_embedding_size * 2, dnn_hidden_units,
                        activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                        init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.add_regularization_loss(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2_reg_dnn)

        self.add_regularization_loss(self.dnn_linear.weight, l2_reg_dnn)
        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns_outof_hist,
                                                                                  self.embedding_dict)
        # query_emb_list
        query_emb_list, _ =  self.input_from_feature_columns(X, self.query_feats, self.embedding_dict)
        # key_emb_list
        keys_emb_list, _ =  self.input_from_feature_columns(X, self.hist_feats, self.embedding_dict)
        keys_emb_list = [emb.squeeze_(dim=1) for emb in keys_emb_list]

        # batch_size, T, embedding_size * field_size
        keys_emb = concat_fun(keys_emb_list)
        # batch_size, 1, embedding_size * field_size
        query_emb = concat_fun(query_emb_list)
        # keys_mask
        hist_var = self.hist_feats[0]
        X_hist = X[:, self.feature_index[hist_var.name][0]:self.feature_index[hist_var.name][1]].long()
        zeros = torch.zeros_like(X_hist)
        keys_mask = torch.where(X_hist > 0, X_hist, zeros).bool()
        keys_mask = keys_mask.unsqueeze(dim=1)

        dense_input = combined_dnn_input(sparse_embedding_list + query_emb_list, dense_value_list)
        hist = self.att_pool([query_emb, keys_emb, keys_mask])

        hist = torch.flatten(hist, start_dim=1)
        deep_input_emb = concat_fun([dense_input, hist])
        dnn_input = torch.flatten(deep_input_emb, start_dim=1)

        output = self.dnn(dnn_input)
        final_logit = self.dnn_linear(output)

        output = self.out(final_logit)

        return output