# -*- coding:utf-8 -*-
"""
Author:
    Yuef Zhang
    Xierry, Xierry@outlook.com
Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
"""


from .basemodel import BaseModel
from ..inputs import DenseFeat, SparseFeat, VarLenSparseFeat, \
    varlen_embedding_lookup, embedding_lookup, get_varlen_pooling_list
from ..layers import *
from ..layers.sequence import AttentionSequencePoolingLayer


class DIN(BaseModel):
    """Instantiates the Deep Interest Network architecture.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_units: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool. Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return:  A PyTorch model instance.
    """

    def __init__(self, dnn_feature_columns, history_feature_list, dnn_use_bn=False,
                 dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_units=(64, 16),
                 att_activation='Dice', att_weight_normalization=False, l2_reg_dnn=0.0, 
                 l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001,
                 seed=1024, task='binary', device='cpu'):
        super(DIN, self).__init__([], dnn_feature_columns,
                                  dnn_hidden_units=dnn_hidden_units, l2_reg_linear=0,
                                  l2_reg_dnn=l2_reg_dnn, init_std=init_std,
                                  l2_reg_embedding=l2_reg_embedding,
                                  dnn_dropout=dnn_dropout, dnn_activation=dnn_activation,
                                  seed=seed, task=task, 
                                  device=device)

        self.ad_feature_columns, self.behavior_feature_columns, self.default_sparse_feature_columns, \
            self.default_varlen_sparse_feature_columns, self.dense_feature_columns, self.behavior_fc_with_length = \
                split_feature_columns(dnn_feature_columns, history_feature_list)

        att_emb_dim = self._compute_interest_dim()
        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_units,
                                                       embedding_dim=att_emb_dim,
                                                       activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=self.behavior_fc_with_length is None,
                                                       weight_normalization=att_weight_normalization)

        self.dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.to(device)

    def forward(self, X):
        # query -> AD , keys -> user_behavior hist
        query_emb_list, keys_emb_list, keys_length_or_mask, dnn_input_emb_list, dense_value_list = \
            self._get_embedding_list(X)

        # concatenate        
        query_emb = torch.cat(query_emb_list, dim=-1)           # -> [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)             # -> [B, T, E]
        dnn_input_emb = torch.cat(dnn_input_emb_list, dim=-1)   # -> [B, 1, E_dnn_0]

        att_interest_hist = self.attention(query_emb, keys_emb, keys_length_or_mask, keys_length_or_mask) # -> [B, 1, E]

        # deep part                                             # -> [B, E_dnn_1]
        dnn_input = torch.cat([att_interest_hist.squeeze(), dnn_input_emb.squeeze()] + dense_value_list, dim=-1)

        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)

        return y_pred

    def _get_embedding_list(self, X):
        #####  Interest Part
        # query -> AD
        query_emb_list = embedding_lookup(
            X, self.embedding_dict, self.feature_index, 
            self.ad_feature_columns, to_list=True)

        # keys -> user_behavior hist
        keys_emb_list = embedding_lookup(
            X, self.embedding_dict, self.feature_index, 
            self.behavior_feature_columns, to_list=True)

        ###### NonInterest dnn Part embedding
        dnn_sparse_emb_list = embedding_lookup(
            X, self.embedding_dict, self.feature_index, 
            self.default_sparse_feature_columns, to_list=True)

        sequence_embed_dict = varlen_embedding_lookup(
            X,  self.embedding_dict, self.feature_index,
            self.default_varlen_sparse_feature_columns)
        dnn_varlen_sparse_emb_list = get_varlen_pooling_list(
            sequence_embed_dict, X, self.feature_index,
            self.default_varlen_sparse_feature_columns, self.device)

        dnn_input_emb_list = query_emb_list + dnn_sparse_emb_list + dnn_varlen_sparse_emb_list

        _, dense_value_list = self.input_from_feature_columns(
            X, self.dense_feature_columns, self.embedding_dict)

        if self.behavior_fc_with_length is None:
            lookup = self.feature_index[self.behavior_feature_columns[0].name]
            keys_length_or_mask = X[:, lookup[0]: lookup[1]] != 0
        else:
            lookup = self.feature_index[self.behavior_fc_with_length.length_name]
            keys_length_or_mask = X[:, lookup[0]: lookup[1]].long()

        return query_emb_list, keys_emb_list, keys_length_or_mask, dnn_input_emb_list, dense_value_list

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.ad_feature_columns:
            interest_dim += feat.embedding_dim
        return interest_dim


def split_feature_columns(feature_columns, history_feature_list):

    ad_feature_columns=[]
    behavior_feature_columns=[]
    default_sparse_feature_columns=[]
    default_varlen_sparse_feature_columns=[]
    dense_feature_columns = []
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            if fc.name in history_feature_list:
                ad_feature_columns.append(fc)
            else:
                default_sparse_feature_columns.append(fc)
        elif isinstance(fc, VarLenSparseFeat):
            if fc.name.startswith('hist') and fc.name.replace("hist_", "") in history_feature_list:
                behavior_feature_columns.append(fc)
            else:
                default_varlen_sparse_feature_columns.append(fc)
        elif isinstance(fc, DenseFeat):
            dense_feature_columns.append(fc)
    
    if len(ad_feature_columns) == 0 or len(behavior_feature_columns) == 0:
        raise ValueError("Goods or History feature columns is None, "
                         "check history_feature_list {} and Interest feature columns {} matched".format(history_feature_list, ad_feature_columns + behavior_feature_columns))
    elif len(ad_feature_columns) != len(behavior_feature_columns):
        raise ValueError("Goods num_feature: {} do not equel History num_feature: {}.".format(len(ad_feature_columns), len(behavior_feature_columns)))
 
    behavior_fc_with_length = [fc for fc in behavior_feature_columns if fc.length_name is not None]
    behavior_fc_with_length = behavior_fc_with_length[0] if len(behavior_fc_with_length) > 0 else None

    return ad_feature_columns, behavior_feature_columns, default_sparse_feature_columns, default_varlen_sparse_feature_columns, dense_feature_columns, behavior_fc_with_length
