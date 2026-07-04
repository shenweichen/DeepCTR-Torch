# -*- coding: utf-8 -*-
import torch
import pytest

from deepctr_torch.models import (
    AutoInt,
    DCN,
    DCNMix,
    DIFM,
    DeepFM,
    FiBiNET,
    IFM,
    NFM,
    ONN,
    WDL,
    xDeepFM,
)
from deepctr_torch.models.din import DIN
from deepctr_torch.layers import DNN, create_linear
from .DIN_test import get_xy_fd
from ..utils import get_test_data


def _registered_l2(model):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    result = {}
    for weights, _, l2_value in model.regularization_weight:
        for item in weights:
            parameter = item[1] if isinstance(item, tuple) else item
            result[names[id(parameter)]] = l2_value
    return result


def _feature_columns():
    return get_test_data(
        16, sparse_feature_num=2, dense_feature_num=1, sequence_feature=[]
    )[2]


def test_compile_uses_mean_loss_and_native_adam():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=1, dense_feature_num=1
    )
    model = DeepFM(feature_columns, feature_columns, device="cpu")
    model.compile("adam", "binary_crossentropy")
    assert isinstance(model.optim, torch.optim.Adam)
    assert model.loss_reduction == "mean"


def test_dnn_uses_glorot_normal_weights_and_zero_bias():
    torch.manual_seed(2026)
    layer = DNN(100, (80,), device="cpu")
    weight = layer.linears[0].weight.detach()
    expected_std = (2.0 / (100 + 80)) ** 0.5
    assert abs(weight.std().item() - expected_std) < expected_std * 0.1
    assert torch.equal(
        layer.linears[0].bias.detach(),
        torch.zeros_like(layer.linears[0].bias),
    )


def test_output_projection_uses_glorot_uniform_and_zero_bias():
    torch.manual_seed(2026)
    layer = create_linear(100, 80, bias=True)
    bound = (6.0 / (100 + 80)) ** 0.5
    assert layer.weight.detach().abs().max().item() <= bound
    assert torch.equal(layer.bias.detach(), torch.zeros_like(layer.bias))


def test_linear_rejects_unknown_initializer():
    with pytest.raises(ValueError, match="initializer"):
        create_linear(4, 2, initializer="unknown")


def test_autoint_exposes_and_applies_linear_l2():
    columns = _feature_columns()
    model = AutoInt(
        columns, columns, dnn_hidden_units=(8,), att_layer_num=1,
        l2_reg_linear=0.11, l2_reg_embedding=0.12, l2_reg_dnn=0.13,
        device="cpu",
    )
    registered = _registered_l2(model)
    assert registered["linear_model.embedding_dict.sparse_feature_0.weight"] == 0.11
    assert registered["embedding_dict.sparse_feature_0.weight"] == 0.12
    assert registered["dnn.linears.0.weight"] == 0.13
    assert "dnn_linear.weight" not in registered


def test_autoint_defaults_match_tf_attention_and_dnn_widths():
    columns = _feature_columns()
    model = AutoInt(columns, columns, device="cpu")
    assert [linear.out_features for linear in model.dnn.linears] == [256, 128, 64]
    assert model.int_layers[0].W_Query.shape == (model.embedding_size, 16)
    assert model.int_layers[1].W_Query.shape == (16, 16)
    assert model.dnn_linear.in_features == 64 + len(model.embedding_dict) * 16


@pytest.mark.parametrize("model_type", [DeepFM, FiBiNET, NFM, ONN, WDL, xDeepFM])
def test_dnn_l2_covers_hidden_kernels_but_not_output(model_type):
    columns = _feature_columns()
    model = model_type(
        columns, columns, dnn_hidden_units=(8,),
        l2_reg_linear=0.11, l2_reg_embedding=0.12, l2_reg_dnn=0.13,
        device="cpu",
    )
    registered = _registered_l2(model)
    assert registered["dnn.linears.0.weight"] == 0.13
    assert "dnn_linear.weight" not in registered


@pytest.mark.parametrize("model_type", [DCN, DCNMix])
def test_dcn_l2_routes_linear_dnn_and_cross_coefficients(model_type):
    columns = _feature_columns()
    model = model_type(
        columns, columns, dnn_hidden_units=(8,), cross_num=1,
        l2_reg_linear=0.11, l2_reg_embedding=0.12,
        l2_reg_dnn=0.13, l2_reg_cross=0.14, device="cpu",
    )
    registered = _registered_l2(model)
    assert registered["linear_model.embedding_dict.sparse_feature_0.weight"] == 0.11
    assert registered["dnn.linears.0.weight"] == 0.13
    assert "dnn_linear.weight" not in registered
    cross_values = [
        value for name, value in registered.items() if name.startswith("crossnet.")
    ]
    assert cross_values and all(value == 0.14 for value in cross_values)


def test_ifm_l2_excludes_unregularized_output_projection():
    columns = _feature_columns()
    model = IFM(columns, columns, dnn_hidden_units=(8,), l2_reg_dnn=0.13)
    registered = _registered_l2(model)
    assert registered["factor_estimating_net.linears.0.weight"] == 0.13
    assert "transform_weight_matrix_P.weight" not in registered


def test_difm_l2_only_covers_the_dnn_hidden_kernels():
    columns = _feature_columns()
    model = DIFM(columns, columns, dnn_hidden_units=(8,), l2_reg_dnn=0.13)
    registered = _registered_l2(model)
    assert registered["bit_wise_net.linears.0.weight"] == 0.13
    assert not any(name.startswith("vector_wise_net.") for name in registered)
    assert "transform_matrix_P_vec.weight" not in registered
    assert "transform_matrix_P_bit.weight" not in registered


def test_difm_defaults_match_tf_attention_and_dnn_widths():
    columns = _feature_columns()
    model = DIFM(columns, columns, device="cpu")
    assert [linear.out_features for linear in model.bit_wise_net.linears] == [256, 128, 64]
    assert model.vector_wise_net.W_Query.shape == (model.embedding_size, 64)
    assert model.transform_matrix_P_vec.in_features == model.sparse_feat_num * 64


def test_din_l2_covers_hidden_kernels_but_not_output():
    _, _, columns, behavior_features = get_xy_fd()
    model = DIN(
        columns, behavior_features, dnn_hidden_units=(8,),
        l2_reg_embedding=0.12, l2_reg_dnn=0.13,
    )
    registered = _registered_l2(model)
    assert registered["dnn.linears.0.weight"] == 0.13
    assert "dnn_linear.weight" not in registered
