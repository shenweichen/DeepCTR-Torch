# -*- coding: utf-8 -*-
import torch

from deepctr_torch.models import DeepFM
from ..utils import get_test_data


def test_compile_uses_mean_loss_and_native_adam():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=1, dense_feature_num=1
    )
    model = DeepFM(feature_columns, feature_columns, device="cpu")
    model.compile("adam", "binary_crossentropy")
    assert isinstance(model.optim, torch.optim.Adam)
    assert model.loss_reduction == "mean"
