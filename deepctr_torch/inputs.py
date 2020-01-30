# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com
"""

from collections import OrderedDict, namedtuple, defaultdict
from itertools import chain

import torch
import torch.nn as nn

from .layers.utils import concat_fun
from .layers.sequence import SequencePoolingLayer

DEFAULT_GROUP_NAME = "default_group"


class SparseFeat(namedtuple('SparseFeat', ['name', 'dimension', 'use_hash', 'dtype', 'embedding_name', 'embedding', 'group_name'])):
    __slots__ = ()

    def __new__(cls, name, dimension, use_hash=False, dtype="int32",
                embedding_name=None, embedding=True, group_name=DEFAULT_GROUP_NAME):
        if embedding and embedding_name is None:
            embedding_name = name
        return super(SparseFeat, cls).__new__(cls, name, dimension, use_hash, dtype, embedding_name, embedding, group_name)


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)


class VarLenSparseFeat(namedtuple('VarLenFeat',
                                  ['name', 'dimension', 'maxlen', 'combiner', 'use_hash', 'dtype', 'embedding_name',
                                   'embedding', 'group_name','length_name'])):
    __slots__ = ()

    def __new__(cls, name, dimension, maxlen, combiner="mean", use_hash=False, dtype="float32", embedding_name=None,
                embedding=True, group_name=DEFAULT_GROUP_NAME,length_name=None):
        if embedding_name is None:
            embedding_name = name
        return super(VarLenSparseFeat, cls).__new__(cls, name, dimension, maxlen, combiner, use_hash, dtype,

                                                    embedding_name, embedding, group_name,length_name)


def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())

def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))


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
        elif isinstance(feat,VarLenSparseFeat):
            features[feat_name] = (start, start + feat.maxlen)
            start += feat.maxlen
            if feat.length_name is not None:
                features[feat.length_name] = (start, start + 1)
                start += 1
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features


def get_dense_input(features, feature_columns):
    dense_feature_columns = list(filter(lambda x: isinstance(
        x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        dense_input_list.append(features[fc.name])
    return dense_input_list


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
