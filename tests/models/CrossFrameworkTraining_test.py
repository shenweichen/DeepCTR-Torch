# -*- coding: utf-8 -*-
import numpy as np
import torch

from deepctr_torch.models import DeepFM
from deepctr_torch.initializers import (
    export_deepfm_parameters,
    load_deepfm_parameters,
)
from ..utils import get_test_data


def test_deepfm_compile_uses_aligned_defaults():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=1, dense_feature_num=1
    )
    model = DeepFM(feature_columns, feature_columns, device="cpu")
    model.compile("adam", "binary_crossentropy")
    assert isinstance(model.optim, torch.optim.Adam)
    assert model.loss_reduction == "mean"


def test_default_initialization_is_semantically_reproducible():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=2, dense_feature_num=2
    )
    first = DeepFM(
        feature_columns, feature_columns, dnn_hidden_units=(8,), seed=2026,
        device="cpu"
    )
    second = DeepFM(
        feature_columns, feature_columns, dnn_hidden_units=(8,), seed=2026,
        device="cpu"
    )
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name


def test_portable_parameters_round_trip_between_models():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=2, dense_feature_num=2
    )
    source = DeepFM(
        feature_columns, feature_columns, dnn_hidden_units=(8,), seed=2026,
        device="cpu"
    )
    target = DeepFM(
        feature_columns, feature_columns, dnn_hidden_units=(8,), seed=2027,
        device="cpu"
    )
    load_deepfm_parameters(target, export_deepfm_parameters(source))
    for name, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[name]), name
