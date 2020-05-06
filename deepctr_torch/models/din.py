# -*- coding:utf-8 -*-
"""
Author:
    Yuef Zhang
    Xierry
Reference:
    [1] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068. (https://arxiv.org/pdf/1706.06978.pdf)
"""

from .basemodel import BaseModel
from ..inputs import *
from ..layers import *
from ..layers.sequence import AttentionSequencePoolingLayer, SequencePoolingLayer


class DIN(BaseModel):
    """Instantiates the Deep Interest Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
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
                 dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_size=(64, 16),
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

        for i in range(len(dnn_feature_columns)):
            fc = dnn_feature_columns[i]
            if hasattr(fc, 'group_name'):
                if fc.name in history_feature_list:
                    dnn_feature_columns[i] = SparseFeat(
                        fc.name, fc.vocabulary_size, fc.embedding_dim, 
                        fc.use_hash, fc.dtype, fc.embedding_name, 'ad_group')
                elif 'hist' in fc.name:
                    dnn_feature_columns[i] = VarLenSparseFeat(SparseFeat(
                        fc.name, fc.vocabulary_size, fc.embedding_dim, 
                        fc.sparsefeat.use_hash, fc.dtype, fc.embedding_name, 'behavior_group'),
                        fc.maxlen, fc.combiner, fc.length_name)
        self.dnn_input = DinInput(dnn_feature_columns)
        self.behavior_fc0 = self.dnn_input.get_varlen_sparse_feature_columns(self.dnn_input.behavior_group)[0]
        att_emb_dim = self._compute_interest_dim()

        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=att_emb_dim,
                                                       activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
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
        query_emb_list, keys_emb_list, keys_length, dnn_input_emb_list, dense_value_list = \
            self._get_embedding_list(X)

        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)           # -> [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)             # -> [B, T, E]
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)  # -> [B, 1, E_dnn_0]

        att_interest_hist = self.attention(query_emb, keys_emb, keys_length) # -> [B, 1, E]

        # deep part                                             # -> [B, E_dnn_1]
        dnn_input = torch.cat([att_interest_hist.squeeze(), deep_input_emb.squeeze()] + dnn_input_emb_list, dim=-1)                               

        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)

        return y_pred

    def _get_embedding_list(self, X):
        ##### Interest Part
        # query -> AD
        query_emb_list = self.input_from_sparse_columns(
            X, self.dnn_input.sparse_group_dict[self.dnn_input.ad_group])

        # keys -> user_behavior hist
        keys_emb_list = self.input_from_varlen_sparse_columns(
            X, self.dnn_input.varlen_sparse_group_dict[self.dnn_input.behavior_group], pooling=False)

        ### NonInterest dnn Part embedding
        dnn_sparse_emb_list = self.input_from_sparse_columns(
            X, self.dnn_input.sparse_group_dict[self.dnn_input.default_group])

        dnn_varlen_sparse_emb_list = self.input_from_varlen_sparse_columns(
            X, self.dnn_input.varlen_sparse_group_dict[self.dnn_input.default_group])

        dnn_input_emb_list = query_emb_list + dnn_sparse_emb_list + dnn_varlen_sparse_emb_list

        dense_value_list = self.input_from_dense_columns(X, self.dnn_input.dense_feature_columns)

        if self.behavior_fc0.length_name is None:
            lookup = self.feature_index[self.behavior_fc0.name]
            keys_mask = X[:, lookup[0]: lookup[1]] != 0
            keys_length = keys_mask.sum(-1, keepdim=True, dtype=torch.long)
        else:
            lookup = self.feature_index[self.behavior_fc0.length_name]
            keys_length = X[:, lookup[0]: lookup[1]].long()

        return query_emb_list, keys_emb_list, keys_length, dnn_input_emb_list, dense_value_list

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.dnn_input.sparse_group_dict[self.dnn_input.ad_group]:
            interest_dim += feat.embedding_dim
        return interest_dim

    def input_from_dense_columns(self, X, feature_columns):

        dense_value_list = [
            X[:, self.feature_index[fc.name][0]: self.feature_index[fc.name][1]] 
            for fc in feature_columns if isinstance(fc, DenseFeat)]

        return dense_value_list

    def input_from_sparse_columns(self, X, feature_columns):

        sparse_embedding_list = [
            self.embedding_dict[fc.embedding_name](
                X[:, self.feature_index[fc.name][0]: self.feature_index[fc.name][1]].long()
            ) for fc in feature_columns if isinstance(fc, SparseFeat)]

        return sparse_embedding_list

    def input_from_varlen_sparse_columns(self, X, feature_columns, pooling=True):

        varlen_sparse_embedding_list = [
            self.get_varlen_sparse_feature_pooling(X, fc) if pooling else 
            self.embedding_dict[fc.embedding_name](
                X[:, self.feature_index[fc.name][0]: self.feature_index[fc.name][1]].long()
            ) for fc in feature_columns if isinstance(fc, VarLenSparseFeat)]

        return varlen_sparse_embedding_list

    def get_varlen_sparse_feature_pooling(self, X, fc):

        x = X[:, self.feature_index[fc.name][0]: self.feature_index[fc.name][1]].long()
        seq_emb = self.embedding_dict[fc.embedding_name](x)
        if fc.length_name is None:
            mask = (x != 0)
            seq_emb_pooling = SequencePoolingLayer(mode=fc.combiner, support_masking=True)(seq_emb, mask)
        else:
            seq_length = X[:, self.feature_index[fc.length_name][0]: self.feature_index[fc.length_name][1]].long()
            seq_emb_pooling = SequencePoolingLayer(mode=fc.combiner)(seq_emb, seq_length)

        return seq_emb_pooling    


class Input(object):
    def __init__(self, feature_columns, **kwargs):
        self.default_group = DEFAULT_GROUP_NAME

        self.sparse_feature_columns, self.varlen_sparse_feature_columns, self.dense_feature_columns = \
            split_feature_by_class(feature_columns)
        self.feature_columns = self.sparse_feature_columns + self.varlen_sparse_feature_columns + self.dense_feature_columns
        self.length = len(self.feature_columns)

        self.group_dict = split_feature_by_group(self.feature_columns)
        self.dense_group_dict = split_feature_by_group(self.dense_feature_columns)
        self.sparse_group_dict = split_feature_by_group(self.sparse_feature_columns)
        self.varlen_sparse_group_dict = split_feature_by_group(self.varlen_sparse_feature_columns)

        self.flag()
    
    def get_sparse_feature_columns(self, group_names=None):
        
        if group_names is None:
            return self.sparse_feature_columns
        if isinstance(group_names, str):
            group_names = [group_names]
        if isinstance(group_names, (list, tuple)) and all(isinstance(name, str) for name in group_names):
            return list(chain.from_iterable(self.sparse_group_dict[group_name] for group_name in group_names))
        else:
            raise TypeError(
                "group_name should be string or strings in list-like, got {}".format(group_names))

    def get_varlen_sparse_feature_columns(self, group_names=None):
        if group_names is None:
            return self.varlen_sparse_feature_columns
        if isinstance(group_names, str):
            group_names = [group_names]      
        if isinstance(group_names, (list, tuple)) and all(isinstance(name, str) for name in group_names):
            return list(chain.from_iterable(self.varlen_sparse_group_dict[group_name] for group_name in group_names))
        else:
            raise TypeError(
                "group_name should be string or strings in list-like, got {}".format(group_names))

    def __len__(self):
        return self.length
    
    def flag(self):
        self.dense = len(self.dense_feature_columns) > 0
        self.sparse = len(self.sparse_feature_columns) > 0
        self.varlen = len(self.varlen_sparse_feature_columns) > 0


class DinInput(Input):
    def __init__(self, feature_columns, behavior_group="behavior_group", ad_group="ad_group", **kwargs):
        super().__init__(feature_columns)
        self.behavior_group=behavior_group
        self.ad_group=ad_group

        self.has_behavior_lenth = any(fc.length_name is not None for fc in self.varlen_sparse_group_dict[self.behavior_group])

        self.ad_group_feature_names = [
            fc.name for fc in self.sparse_feature_columns if "ad" in fc.group_name]
        self.behavior_group_feature_names = [
            fc.name for fc in self.sparse_feature_columns if "behavior" in fc.group_name]


def split_feature_by_group(feature_columns):
    group_dict = defaultdict(list)
    for fc in feature_columns:
        if not hasattr(fc, "group_name"):
            continue
        group_dict[fc.group_name].append(fc)

    return group_dict


def split_feature_by_class(feature_columns):
    
    dense_feature_dict = OrderedDict()
    sparse_feature_dict = OrderedDict()
    varlen_sparse_feature_dict = OrderedDict()

    feature_dict_list = [sparse_feature_dict, varlen_sparse_feature_dict, dense_feature_dict]

    for fc in feature_columns:
        f_name = fc.name
        if any(f_name in feature_dict for feature_dict in feature_dict_list):
            continue
        if isinstance(fc, SparseFeat):
            sparse_feature_dict[f_name] = fc
        elif isinstance(fc, VarLenSparseFeat):
            varlen_sparse_feature_dict[f_name] = fc
        elif isinstance(fc, DenseFeat):
            dense_feature_dict[f_name] = fc
        else:
            raise TypeError("Invalid feature type, got {}".format(type(fc)))
    
    sparse_feature_columns = list(sparse_feature_dict.values())
    varlen_sparse_feature_columns = list(varlen_sparse_feature_dict.values())
    dense_feature_columns = list(dense_feature_dict.values())

    return sparse_feature_columns, varlen_sparse_feature_columns, dense_feature_columns
