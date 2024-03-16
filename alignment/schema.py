from dataclasses import dataclass
from torch import Tensor
from transformers.modeling_outputs import ModelOutput

# For Constrastive Alignment Output
@dataclass
class ContrastiveAlignmentOutput(ModelOutput):
    ctr_model_reps: Tensor = None
    text_model_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None

@dataclass
class MlmAlignmentOutput(ModelOutput):
    loss: Tensor = None
