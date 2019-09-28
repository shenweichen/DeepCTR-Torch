# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com
"""

from collections import OrderedDict, namedtuple
from itertools import chain

import torch

from .layers.utils import concat_fun


class SparseFeat(namedtuple('SparseFeat', ['name', 'dimension', 'use_hash', 'dtype', 'embedding_name', 'embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, use_hash=False, dtype="int32", embedding_name=None, embedding=True):
        if embedding and embedding_name is None:
            embedding_name = name
        return super(SparseFeat, cls).__new__(cls, name, dimension, use_hash, dtype, embedding_name, embedding)


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)


class VarLenSparseFeat(namedtuple('VarLenFeat',
                                  ['name', 'dimension', 'maxlen', 'combiner', 'use_hash', 'dtype', 'embedding_name',
                                   'embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, maxlen, combiner="mean", use_hash=False, dtype="float32", embedding_name=None,
                embedding=True):
        if embedding_name is None:
            embedding_name = name
        return super(VarLenSparseFeat, cls).__new__(cls, name, dimension, maxlen, combiner, use_hash, dtype,
                                                    embedding_name, embedding)


def get_fixlen_feature_names(feature_columns):
    features = build_input_features(
        feature_columns, include_varlen=False, include_fixlen=True)
    return list(features.keys())


def get_varlen_feature_names(feature_columns):
    features = build_input_features(
        feature_columns, include_varlen=True, include_fixlen=False)
    return list(features.keys())


def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))


def build_input_features(feature_columns, include_varlen=True, mask_zero=True, prefix='', include_fixlen=True):
    input_features = OrderedDict()
    features = OrderedDict()

    start = 0

    if include_fixlen:
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
    if include_varlen:
        for feat in feature_columns:
            feat_name = feat.name
            if feat_name in features:
                continue
            if isinstance(feat, VarLenSparseFeat):
                features[feat_name] = (start, start + feat.maxlen)
                start += feat.maxlen

    # if include_fixlen:
    #     for fc in feature_columns:
    #         if isinstance(fc, SparseFeat):
    #             input_features[fc.name] = 1
    #             # Input( shape=(1,), name=prefix+fc.name, dtype=fc.dtype)
    #         elif isinstance(fc, DenseFeat):
    #             input_features[fc.name] = 1
    #             # Input(
    #             # shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
    # if include_varlen:
    #     for fc in feature_columns:
    #         if isinstance(fc, VarLenSparseFeat):
    #             input_features[fc.name] = 1
    #             # Input(shape=(fc.maxlen,), name=prefix + 'seq_' + fc.name,
    #             #                                   dtype=fc.dtype)
    #     if not mask_zero:
    #         for fc in feature_columns:
    #             input_features[fc.name + "_seq_length"] = 1
    #             # Input(shape=(
    #             # 1,), name=prefix + 'seq_length_' + fc.name)
    #             input_features[fc.name + "_seq_max_length"] = 1  # fc.maxlen

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