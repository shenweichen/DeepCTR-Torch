# -*- coding: utf-8 -*-
import math

import numpy as np
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


def test_default_adam_matches_tensorflow_reference():
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


def test_default_adam_does_not_update_absent_embedding_rows():
    parameter = torch.nn.Parameter(torch.zeros((3, 1)))
    optimizer = TensorFlowAdam([("embedding_dict.item.weight", parameter)])
    parameter.grad = torch.tensor([[1.0], [0.0], [0.0]])
    optimizer.step()
    first_step_row = parameter.detach()[0].clone()
    parameter.grad = torch.tensor([[0.0], [1.0], [0.0]])
    optimizer.step()
    assert torch.equal(parameter.detach()[0], first_step_row)
    assert not torch.equal(parameter.detach()[1], torch.zeros(1))


def test_deepfm_compile_uses_aligned_defaults():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=1, dense_feature_num=1
    )
    model = DeepFM(feature_columns, feature_columns, device="cpu")
    model.compile("adam", "binary_crossentropy")
    assert isinstance(model.optim, TensorFlowAdam)
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
