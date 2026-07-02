# -*- coding: utf-8 -*-
import math

import numpy as np
import pytest
import torch

from deepctr_torch.models import DeepFM
from deepctr_torch.initializers import (
    export_deepfm_parameters,
    load_deepfm_parameters,
)
from deepctr_torch.optimizers import TensorFlowAdam
from ..utils import get_test_data


def _tensorflow_adam_step(parameter, first, second, gradient, step,
                          learning_rate=1e-3, beta1=0.9, beta2=0.999,
                          epsilon=1e-7):
    first = beta1 * first + (1.0 - beta1) * gradient
    second = beta2 * second + (1.0 - beta2) * gradient * gradient
    step_size = learning_rate * math.sqrt(1.0 - beta2 ** step) / (1.0 - beta1 ** step)
    parameter = parameter - step_size * first / (np.sqrt(second) + epsilon)
    return parameter, first, second


def test_tensorflow_adam_matches_two_step_reference():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = TensorFlowAdam([("weight", parameter)])
    expected = parameter.detach().numpy().copy()
    first = np.zeros_like(expected)
    second = np.zeros_like(expected)
    for step, gradient in enumerate(
            (np.array([0.2, -0.3]), np.array([-0.1, 0.4])), start=1):
        parameter.grad = torch.tensor(gradient, dtype=parameter.dtype)
        optimizer.step()
        expected, first, second = _tensorflow_adam_step(
            expected, first, second, gradient, step
        )
    np.testing.assert_allclose(parameter.detach().numpy(), expected, atol=1e-7)


def test_tensorflow_adam_does_not_update_absent_embedding_rows():
    parameter = torch.nn.Parameter(torch.zeros((3, 1)))
    optimizer = TensorFlowAdam([("embedding_dict.item.weight", parameter)])
    parameter.grad = torch.tensor([[1.0], [0.0], [0.0]])
    optimizer.step()
    first_step_row = parameter.detach()[0].clone()
    parameter.grad = torch.tensor([[0.0], [1.0], [0.0]])
    optimizer.step()
    assert torch.equal(parameter.detach()[0], first_step_row)
    assert not torch.equal(parameter.detach()[1], torch.zeros(1))


def _trained_predictions(x, y, feature_columns):
    model = DeepFM(
        feature_columns,
        feature_columns,
        dnn_hidden_units=(8,),
        dnn_dropout=0.0,
        l2_reg_linear=0.0,
        l2_reg_embedding=0.0,
        l2_reg_dnn=0.0,
        seed=2026,
        device="cpu",
    )
    model.compile("adam", "binary_crossentropy")
    assert isinstance(model.optim, TensorFlowAdam)
    assert model.loss_reduction == "mean"
    model.fit(
        x, y, batch_size=16, epochs=2, verbose=0,
        shuffle=True, shuffle_seed=2026,
    )
    return model.predict(x, batch_size=32)


def test_cross_framework_profile_is_reproducible():
    x, y, feature_columns = get_test_data(
        96, sparse_feature_num=2, dense_feature_num=2
    )
    first = _trained_predictions(x, y, feature_columns)
    second = _trained_predictions(x, y, feature_columns)
    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)


def test_deepfm_defaults_to_cross_framework_semantics():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=1, dense_feature_num=1
    )
    model = DeepFM(feature_columns, feature_columns, device="cpu")
    model.compile("adam", "binary_crossentropy")
    assert model.initialization_profile == "cross_framework"
    assert model.training_profile == "cross_framework"
    assert isinstance(model.optim, TensorFlowAdam)
    assert model.loss_reduction == "mean"


def test_native_profile_preserves_historical_behavior():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=1, dense_feature_num=1
    )
    model = DeepFM(
        feature_columns, feature_columns, device="cpu",
        initialization_profile="native",
    )
    model.compile("adam", "binary_crossentropy")
    assert model.training_profile == "native"
    assert isinstance(model.optim, torch.optim.Adam)
    assert model.loss_reduction == "sum"


def test_portable_initialization_is_semantically_reproducible():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=2, dense_feature_num=2
    )
    first = DeepFM(
        feature_columns, feature_columns, dnn_hidden_units=(8,), seed=2026,
        device="cpu", initialization_profile="cross_framework"
    )
    second = DeepFM(
        feature_columns, feature_columns, dnn_hidden_units=(8,), seed=2026,
        device="cpu", initialization_profile="cross_framework"
    )
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name


def test_portable_parameters_round_trip_between_models():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=2, dense_feature_num=2
    )
    source = DeepFM(
        feature_columns, feature_columns, dnn_hidden_units=(8,), seed=2026,
        device="cpu", initialization_profile="cross_framework"
    )
    target = DeepFM(
        feature_columns, feature_columns, dnn_hidden_units=(8,), seed=2027,
        device="cpu"
    )
    load_deepfm_parameters(target, export_deepfm_parameters(source))
    for name, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[name]), name


def test_unknown_initialization_profile_is_rejected():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=1, dense_feature_num=1
    )
    with pytest.raises(ValueError, match="initialization_profile"):
        DeepFM(
            feature_columns, feature_columns, device="cpu",
            initialization_profile="unknown"
        )


def test_unknown_training_profile_is_rejected():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=1, dense_feature_num=1
    )
    model = DeepFM(feature_columns, feature_columns, device="cpu")
    with pytest.raises(ValueError, match="training_profile"):
        model.compile("adam", "binary_crossentropy", training_profile="unknown")
