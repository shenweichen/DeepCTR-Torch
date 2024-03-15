import os
import random
import json
from dataclasses import dataclass
from typing import Union, List
import itertools
import numpy as np

import datasets
import torch.utils.data as Data
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from .arguments import DataArguments
from ..deepctr_torch.inputs import build_input_features, get_feature_names
from ..deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat)

import logging
logger = logging.getLogger(__name__)


class AlignmentDataset(Dataset):
    """
    对齐任务的数据加载器
    """
    def __init__(
            self,
            data_args: DataArguments,
            tokenizer: PreTrainedTokenizer,
    ):
        self.tok = tokenizer
        self.data_args = data_args


        self.feature_config = json.load(data_args.feature_config_file)
        self.df = pd.read_csv(data_args.train_file)

        # construct nlp model input
        self.text_feature_fields = self.feature_config["text_feature_list"] # ["gender", "age", "occupation"]
        # list of string
        self.train_data_text = []
        for _, row in self.df.iterrows():
            # same as paper setting https://arxiv.org/pdf/2310.09234.pdf
            text = " ".join(["{} is {}".format(feat, row[feat]) for feat in self.text_feature_fields])
            self.train_data_text.append(text)

        # construct ctr model input
        self.sparse_features = self.feature_config["sparse_feature_list"] # ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
        self.target = self.feature_config["label"] # rating
        # feature value tranform
        for feat in self.sparse_features:
            lbe = LabelEncoder()
            self.df[feat] = lbe.fit_transform(self.df[feat])
        # feature columns construct
        fixlen_feature_columns = [SparseFeat(feat, self.df[feat].nunique()) for feat in self.sparse_features]
        # for DeepFM input
        self.linear_feature_columns = fixlen_feature_columns
        self.dnn_feature_columns = fixlen_feature_columns
        feature_names = get_feature_names(self.linear_feature_columns + self.dnn_feature_columns)
        # ctr_model input: list of list
        self.train_data_ctr = np.array([self.df[name].values.tolist() for name in feature_names]).T.tolist()

        # must correspond
        assert len(self.train_data_text) == len(self.train_data_ctr)

        self.total_len = len(self.train_data_ctr)

    def create_one_example(self, text_encoding: List[int]):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.prompt_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        # text model input
        text_feature = self.train_data_text[item]
        text_model_input = self.create_one_example(text_feature)
        # ctr model input
        ctr_model_input = self.train_data_ctr[item]
        return {"text_model_input" : text_model_input,
                "ctr_model_input" : ctr_model_input}


@dataclass
class AlignmentCollator(DataCollatorWithPadding):
    """
    对齐任务的数据收集器
    """

    prompt_max_len: int = 64

    def __call__(self, features):

        # batch inputs for PLM\LLM
        text_input = [feat_map["text_model_input"] for feat_map in features]
        text_input_batch = self.tokenizer.pad(
            text_input,
            padding='max_length',
            max_length=self.prompt_max_len,
            return_tensors="pt",
        )
        # batch inputs for ID Model
        ctr_input_batch = [feat_map["ctr_model_input"] for feat_map in features]
        ctr_input_batch = Data.TensorDataset(ctr_input_batch)
        return ctr_input_batch, text_input_batch
