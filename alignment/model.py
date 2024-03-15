import sys
sys.path.insert(0, '..')

import json
import os
import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributed as dist

from transformers import AutoModel, PreTrainedModel, AutoConfig
from transformers.modeling_outputs import ModelOutput


from typing import Optional, Dict

from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.models.din import DIN
from .arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments

import logging

logger = logging.getLogger(__name__)


@dataclass
class AlignmentOutput(ModelOutput):
    ctr_model_reps: Tensor = None
    text_model_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None

# 最后一层表征维度变换
class LinearPooler(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768,
    ):
        super(LinearPooler, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self._config = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, hidden_states: Tensor = NotImplementedError):
        return self.linear(hidden_states)



class AlignmentModel(nn.Module):
    def __init__(
            self,
            ctr_model: BaseModel,
            text_model: PreTrainedModel,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.ctr_model = ctr_model
        self.text_model = text_model

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        # 判断是否需要维度对齐
        text_model_config = AutoConfig.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        if self.model_args.ctr_hidden_dim == text_model_config.hidden_size:
            logger.info("CTR hidden size equal to Text model hidden size")
            self.add_pooler = False
        else:
            logger.warning("CTR hidden size not equal to Text model hidden size, add pooler layer")
            self.add_pooler = True
            self.pooler = LinearPooler(input_dim=text_model_config.hidden_size,
                                       output_dim=self.model_args.ctr_hidden_dim)
            

    def forward(
            self,
            ctr_feat,
            text_feat
    ):

        ctr_model_reps = self.encode_ctr(ctr_feat)
        text_model_reps = self.encode_text(text_feat)
        effective_bsz = self.train_args.per_device_train_batch_size

        scores = torch.matmul(ctr_model_reps, text_model_reps.transpose(0, 1))
        scores = scores.view(effective_bsz, -1)

        target = torch.arange(
            scores.size(0),
            device=scores.device,
            dtype=torch.long
        )
        loss = self.cross_entropy(scores, target)
        return AlignmentOutput(
            loss=loss,
            scores=scores,
            ctr_model_reps=ctr_model_reps,
            text_model_reps=text_model_reps
        )


    # 文本模型表示
    def encode_text(self, text_feat):
        if text_feat is None:
            return None, None
        text_out = self.text_model(**text_feat, return_dict=True, output_hidden_states=True)
        text_hidden = text_out.hidden_states[-1]
        text_reps = text_hidden[:, 0]
        if self.add_pooler:
            text_reps = self.pooler(text_reps)
        return text_reps

    # CTR模型表示
    def encode_ctr(self, ctr_feat):
        if id is None:
            return None, None
        result_dict = self.ctr_model.forward_for_alignment(ctr_feat)
        ctr_reps = result_dict["hidden_state"]
        return ctr_reps
    
    # 权重保存
    def save(self, output_dir):
        # 保存对齐后的CTR模型权重
        ctr_model_save_path = os.makedirs(os.path.join(output_dir, 'ctr_model'))
        torch.save(self.ctr_model.state_dict(), ctr_model_save_path)
