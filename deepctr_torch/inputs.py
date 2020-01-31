# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com
"""

from collections import OrderedDict, namedtuple

import torch
import torch.nn as nn

from .layers.sequence import SequencePoolingLayer
from .layers.utils import concat_fun

DEFAULT_GROUP_NAME = "default_group"


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name',
                             'group_name'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())


# def get_inputs_list(inputs):
#     return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))


def build_input_features(feature_columns):
    # Return OrderedDict: {feature_name:(start, start+dimension)}

    features = OrderedDict()

    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
        elif isinstance(feat, VarLenSparseFeat):
            features[feat_name] = (start, start + feat.maxlen)
            start += feat.maxlen
            if feat.length_name is not None:
                features[feat.length_name] = (start, start + 1)
                start += 1
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features


# def get_dense_input(features, feature_columns):
#     dense_feature_columns = list(filter(lambda x: isinstance(
#         x, DenseFeat), feature_columns)) if feature_columns else []
#     dense_input_list = []
#     for fc in dense_feature_columns:
#         dense_input_list.append(features[fc.name])
#     return dense_input_list


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError

    #
    # def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
    #                      mask_feat_list=(), to_list=False):
    #     """
    #         Args:
    #             sparse_embedding_dict: nn.ModuleDict, {embedding_name: nn.Embedding}
    #             sparse_input_dict: OrderedDict, {feature_name:(start, start+dimension)}
    #             sparse_feature_columns: list, sparse features
    #             return_feat_list: list, names of feature to be returned, defualt () -> return all features
    #             mask_feat_list, list, names of feature to be masked in hash transform
    #         Return:
    #             group_embedding_dict: defaultdict(list)
    #     """
    #     group_embedding_dict = defaultdict(list)
    #     for fc in sparse_feature_columns:
    #         feature_name = fc.name
    #         embedding_name = fc.embedding_name
    #         if (len(return_feat_list) == 0 or feature_name in return_feat_list):
    #             if fc.use_hash:
    #                 # lookup_idx = Hash(fc.vocabulary_size, mask_zero=(feature_name in mask_feat_list))(
    #                 #     sparse_input_dict[feature_name])
    #                 # TODO: add hash function
    #                 lookup_idx = sparse_input_dict[feature_name]
    #             else:
    #                 lookup_idx = sparse_input_dict[feature_name]
    #
    #             group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    #     if to_list:
    #         return list(chain.from_iterable(group_embedding_dict.values()))
    #     return group_embedding_dict
    #
    #
    # def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    #     varlen_embedding_vec_dict = {}
    #     for fc in varlen_sparse_feature_columns:
    #         feature_name = fc.name
    #         embedding_name = fc.embedding_name
    #         if fc.use_hash:
    #             # lookup_idx = Hash(fc.vocabulary_size, mask_zero=True)(sequence_input_dict[feature_name])
    #             # TODO: add hash function
    #             lookup_idx = sequence_input_dict[feature_name]
    #         else:
    #             lookup_idx = sequence_input_dict[feature_name]
    #         varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    #     return varlen_embedding_vec_dict
    #
    #
    # def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns, to_list=False):
    #     pooling_vec_list = defaultdict(list)
    #     for fc in varlen_sparse_feature_columns:
    #         feature_name = fc.name
    #         combiner = fc.combiner
    #         feature_length_name = fc.length_name
    #         if feature_length_name is not None:
    #             seq_input = embedding_dict[feature_name]
    #             vec = SequencePoolingLayer(combiner)([seq_input, features[feature_length_name]])
    #         else:
    #             seq_input = embedding_dict[feature_name]
    #             vec = SequencePoolingLayer(combiner)(seq_input)
    #         pooling_vec_list[fc.group_name].append(vec)
    #
    #         if to_list:
    #             return chain.from_iterable(pooling_vec_list.values())
    #
    #     return pooling_vec_list


def get_varlen_pooling_list(embedding_dict, features, feature_index, varlen_sparse_feature_columns, device):
    varlen_sparse_embedding_list = []

    for feat in varlen_sparse_feature_columns:
        seq_emb = embedding_dict[feat.embedding_name](
            features[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long())
        if feat.length_name is None:
            seq_mask = features[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long() != 0

            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=True, device=device)(
                [seq_emb, seq_mask])
        else:
            seq_length = features[:,
                         feature_index[feat.length_name][0]:feature_index[feat.length_name][1]].long()
            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=False, device=device)(
                [seq_emb, seq_length])
        varlen_sparse_embedding_list.append(emb)
    return varlen_sparse_embedding_list


def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse)
         for feat in
         sparse_feature_columns + varlen_sparse_feature_columns}
    )

    # for feat in varlen_sparse_feature_columns:
    #     embedding_dict[feat.embedding_name] = nn.EmbeddingBag(
    #         feat.dimension, embedding_size, sparse=sparse, mode=feat.combiner)

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict.to(device)
