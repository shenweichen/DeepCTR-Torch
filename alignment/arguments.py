import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )
    fine_tuning: bool = field(
        default=False,
        metadata={"help": "whether to fix plm parameters"}
    )

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=1)

    # p*-tuning
    model_type: str = field(
        default="bert",
        metadata={
            "help": "The type of model, where we currently support bert, roberta, deberta"
        }
    )
    prefix: bool = field(
        default=False,
        metadata={
            "help": "Will use P-tuning v2 during training"
        }
    )
    prompt: bool = field(
        default=False,
        metadata={
            "help": "Will use prompt tuning during training"
        }
    )
    prompt_from_vocab: bool = field(
        default=True,
        metadata={
            "help": "Will prompt embeddings initalized from plm's word embeddings"
        }
    )
    prompt_encoder_type: str = field(default=None)
    pre_seq_len: int = field(
        default=100,
        metadata={
            "help": "The length of prompt"
        }
    )
    prefix_projection: bool = field(
        default=False,
        metadata={
            "help": "Apply a two-layer MLP head over the prefix embeddings"
        }
    )
    prefix_hidden_size: int = field(
        default=512,
        metadata={
            "help": "The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used"
        }
    )
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={
            "help": "The dropout probability used in the models"
        }
    )

    ctr_hidden_dim: int = field(
        default=128,
        metadata={
            "help": "The hidden size of the CTR model last represente layer output"
        }
    )


@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )

    train_file: str = field(
        default=None, metadata={"help": "the whole data for train & eval & test"}
    )

    feature_config_file: str = field(
        default=None, metadata={"help": "feature config file path for ctr model"}
    )

    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"})


    encode_in_path: str = field(default=None, metadata={"help": "Path to data to encode"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    def __post_init__(self):
        if self.dataset_name is not None:
            info = self.dataset_name.split('/')
            self.dataset_split = info[-1] if len(info) == 3 else 'train'
            self.dataset_name = "/".join(info[:-1]) if len(info) == 3 else '/'.join(info)
            self.dataset_language = 'default'
            if ':' in self.dataset_name:
                self.dataset_name, self.dataset_language = self.dataset_name.split(':')
        if self.train_dir is not None:
            files = os.listdir(self.train_dir)
            self.train_path = [
                os.path.join(self.train_dir, f)
                for f in files
                if f.endswith('tsv') or f.endswith('json')
            ]


@dataclass
class DenseTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)
