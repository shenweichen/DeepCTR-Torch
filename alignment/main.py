import logging
import os
import sys
sys.path.insert(0, '..')

from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import (
    HfArgumentParser,
    set_seed,
)

from .arguments import ModelArguments, DataArguments, \
    AlignmentTrainingArguments as TrainingArguments
from .data import AlignmentDataset, ContrastiveAlignmentCollator, MlmAlignmentCollator
from .model import ContrastiveAlignmentModel, MlmAlignmentModel
from .trainer import AlignmentTrainer as Trainer

from deepctr_torch.models.deepfm import DeepFM

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    # build dataset
    train_dataset = AlignmentDataset(
            data_args, tokenizer
    )

    # build CTR model
    ctr_model = DeepFM(train_dataset.linear_feature_columns, 
                   train_dataset.dnn_feature_columns, 
                   task='regression')

    # build text model    
    text_model = AutoModel.from_pretrained(model_args.model_name_or_path, add_pooling_layer=False)

    # build alignment train model
    if training_args.alignment_mode == "contrastive":
        alignment_model = ContrastiveAlignmentModel(ctr_model = ctr_model,
                                        text_model = text_model,
                                        model_args = model_args,
                                        data_args = data_args,
                                        train_args = training_args)
        data_collator=ContrastiveAlignmentCollator(tokenizer=tokenizer)

    elif training_args.alignment_mode == "mlm":
        alignment_model = MlmAlignmentModel(ctr_model = ctr_model,
                                        text_model = text_model,
                                        model_args = model_args,
                                        data_args = data_args,
                                        train_args = training_args)
        data_collator=MlmAlignmentCollator(tokenizer=tokenizer)
    else:
        raise ValueError("Alignment mode must be in [contrastive, mlm]")

    trainer = Trainer(
        model=alignment_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
