import json
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


from .arguments import DataArguments
from ..deepctr_torch.inputs import get_feature_names
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


# data collector for contrastive alignment
@dataclass
class ContrastiveAlignmentCollator(DataCollatorWithPadding):

    tokenizer: PreTrainedTokenizerBase
    max_len: int = 64

    def __call__(self, features):

        # batch inputs for nlp model
        text_input = [feat_map["text_model_input"] for feat_map in features]
        text_input_batch = self.tokenizer.pad(
            text_input,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        # batch inputs for ID Model
        ctr_input_batch = [feat_map["ctr_model_input"] for feat_map in features]
        ctr_input_batch = Data.TensorDataset(ctr_input_batch)
        return ctr_input_batch, text_input_batch


# data collector for mask language modeling alignment
@dataclass
class MlmAlignmentCollator(DataCollatorWithPadding):

    tokenizer: PreTrainedTokenizerBase
    max_len: Optional[int] = 64
    mlm_probability: float = 0.15

    def __call__(self, features):

        # batch inputs for nlp model
        text_input = [feat_map["text_model_input"] for feat_map in features]
        batch = self.tokenizer.pad(
            text_input,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt",
        )
        # generate input & label for mlm train
        batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])
        # batch inputs for CTR Model
        ctr_input_batch = [feat_map["ctr_model_input"] for feat_map in features]
        batch["ctr_input_ids"] = Data.TensorDataset(ctr_input_batch)
        return batch
    
    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
