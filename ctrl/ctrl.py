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

from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import ModelOutput


from typing import Optional, Dict

from deepctr_torch.models.basemodel import BaseModel
from .arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
from .modeling_sequence_classification import BertPrefixForSequenceClassification, BertPromptForSequenceClassification, RobertaPrefixForSequenceClassification, RobertaPromptForSequenceClassification, DebertaPrefixForSequenceClassification
from transformers import BertForSequenceClassification, RobertaForSequenceClassification

import logging

logger = logging.getLogger(__name__)


@dataclass
class DenseOutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class LinearPooler(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768,
            tied=True
    ):
        super(LinearPooler, self).__init__()
        self.linear_q = nn.Linear(input_dim, output_dim)
        if tied:
            self.linear_p = self.linear_q
        else:
            self.linear_p = nn.Linear(input_dim, output_dim)

        self._config = {'input_dim': input_dim, 'output_dim': output_dim, 'tied': tied}

    def forward(self, q: Tensor = None, p: Tensor = None):
        if q is not None:
            return self.linear_q(q[:, 0])
        elif p is not None:
            return self.linear_p(p[:, 0])
        else:
            raise ValueError

    def load(self, ckpt_dir: str):
        if ckpt_dir is not None:
            _pooler_path = os.path.join(ckpt_dir, 'pooler.pt')
            if os.path.exists(_pooler_path):
                logger.info(f'Loading Pooler from {ckpt_dir}')
                state_dict = torch.load(os.path.join(ckpt_dir, 'pooler.pt'), map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


class CtrlModel(nn.Module):
    def __init__(
            self,
            ctr_model: BaseModel,
            lm_p: PreTrainedModel,
            pooler: nn.Module = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

        if train_args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
    ):

        q_hidden, q_reps = self.encode_query(query)
        p_hidden, p_reps = self.encode_passage(passage)

        if q_reps is None or p_reps is None:
            return DenseOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        if self.training:
            if self.train_args.negatives_x_device:
                q_reps = self.dist_gather_tensor(q_reps)
                p_reps = self.dist_gather_tensor(p_reps)

            effective_bsz = self.train_args.per_device_train_batch_size * self.world_size \
                if self.train_args.negatives_x_device \
                else self.train_args.per_device_train_batch_size

            scores = torch.matmul(q_reps, p_reps.transpose(0, 1))
            scores = scores.view(effective_bsz, -1)

            target = torch.arange(
                scores.size(0),
                device=scores.device,
                dtype=torch.long
            )
            target = target * self.data_args.train_n_passages
            loss = self.cross_entropy(scores, target)
            if self.train_args.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
            return DenseOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps
            )

        else:
            loss = None
            if query and passage:
                scores = (q_reps * p_reps).sum(1)
            else:
                scores = None

            return DenseOutput(
                loss=loss,
                scores=scores,
                q_reps=q_reps,
                p_reps=p_reps
            )

    def encode_passage(self, psg):
        if psg is None:
            return None, None

        psg_out = self.lm_p(**psg, return_dict=True, output_hidden_states=True)
        p_hidden = psg_out.hidden_states[-1]
        p_reps = p_hidden[:, 0]
        return p_hidden, p_reps

    def encode_query(self, qry):
        if qry is None:
            return None, None
        qry_out = self.lm_q(**qry, return_dict=True, output_hidden_states=True)
        q_hidden = qry_out.hidden_states[-1]
        q_reps = q_hidden[:, 0]
        return q_hidden, q_reps

    @staticmethod
    def build_pooler(model_args):
        pooler = LinearPooler(
            model_args.projection_in_dim,
            model_args.projection_out_dim,
            tied=not model_args.untie_encoder
        )
        pooler.load(model_args.model_name_or_path)
        return pooler

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        if model_args.model_type == "bert":
            if model_args.prefix:
                model_cls = BertPrefixForSequenceClassification
            elif model_args.prompt:
                model_cls = BertPromptForSequenceClassification
            else:
                model_cls = BertForSequenceClassification
        elif model_args.model_type == "roberta":
            if model_args.prefix:
                model_cls = RobertaPrefixForSequenceClassification
            elif model_args.prompt:
                model_cls = RobertaPromptForSequenceClassification
            else:
                model_cls = RobertaForSequenceClassification
        elif model_args.model_type == "deberta":
            if model_args.prefix:
                model_cls = DebertaPrefixForSequenceClassification
            else:
                raise ValueError("Only support prefix-tuninig for deberta")
        else:
            raise ValueError("Currently only support model_type in 'bert', 'roberta', 'deberta'")

        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = model_cls.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = model_cls.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                lm_q = model_cls.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = model_cls.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        assert not model_args.add_pooler, "We don't support add_pooler because we've chosen sequence classification header as the projection head"
        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args
        )
        return model

    def save(self, output_dir: str):
        if self.model_args.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
        else:
            self.lm_q.save_pretrained(output_dir)

        if self.model_args.add_pooler:
            self.pooler.save_pooler(output_dir)

    def dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors


class DenseModelForInference(CtrlModel):
    POOLER_CLS = LinearPooler

    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            pooler: nn.Module = None,
            **kwargs,
    ):
        nn.Module.__init__(self)
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.pooler = pooler

    @torch.no_grad()
    def encode_passage(self, psg):
        return super(DenseModelForInference, self).encode_passage(psg)

    @torch.no_grad()
    def encode_query(self, qry):
        return super(DenseModelForInference, self).encode_query(qry)

    def forward(
            self,
            query: Dict[str, Tensor] = None,
            passage: Dict[str, Tensor] = None,
    ):
        q_hidden, q_reps = self.encode_query(query)
        p_hidden, p_reps = self.encode_passage(passage)
        return DenseOutput(q_reps=q_reps, p_reps=p_reps)

    @classmethod
    def build(
            cls,
            model_name_or_path: str = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
            **hf_kwargs,
    ):
        assert model_name_or_path is not None or model_args is not None
        assert "config" in hf_kwargs
        if model_name_or_path is None:
            model_name_or_path = model_args.model_name_or_path

        config = hf_kwargs["config"]
        if config.model_type == "bert":
            if config.prefix:
                model_cls = BertPrefixForSequenceClassification
            elif config.prompt:
                model_cls = BertPromptForSequenceClassification
            else:
                model_cls = BertForSequenceClassification
        elif config.model_type == "roberta":
            if config.prefix:
                model_cls = RobertaPrefixForSequenceClassification
            elif config.prompt:
                model_cls = RobertaPromptForSequenceClassification
            else:
                model_cls = RobertaForSequenceClassification
        elif config.model_type == "deberta":
            if config.prefix:
                model_cls = DebertaPrefixForSequenceClassification
            else:
                raise ValueError("Only support prefix-tuninig for deberta")
        else:
            raise ValueError("Currently only support model_type in 'bert', 'roberta', 'deberta'")

        # load local
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            if os.path.exists(_qry_model_path):
                logger.info(f'found separate weight for query/passage encoders')
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = model_cls.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = model_cls.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                lm_q = model_cls.from_pretrained(model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            lm_q = model_cls.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.POOLER_CLS(**pooler_config_dict)
            pooler.load(model_name_or_path)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler
        )
        return model
