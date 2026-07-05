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


def test_training_defaults_match_tf():
    _, _, feature_columns = get_test_data(
        16, sparse_feature_num=1, dense_feature_num=1
    )
    model = DeepFM(feature_columns, feature_columns, device="cpu")
    model.compile("adam", "binary_crossentropy")
    assert isinstance(model.optim, torch.optim.Adam)
    assert model.loss_reduction == "mean"

    torch.manual_seed(2026)
    dnn = DNN(100, (80,), device="cpu")
    weight = dnn.linears[0].weight.detach()
    expected_std = (2.0 / (100 + 80)) ** 0.5
    assert abs(weight.std().item() - expected_std) < expected_std * 0.1
    assert torch.equal(
        dnn.linears[0].bias.detach(),
        torch.zeros_like(dnn.linears[0].bias),
    )

    projection = create_linear(100, 80, bias=True)
    bound = (6.0 / (100 + 80)) ** 0.5
    assert projection.weight.detach().abs().max().item() <= bound
    assert torch.equal(
        projection.bias.detach(), torch.zeros_like(projection.bias))
    with pytest.raises(ValueError, match="initializer"):
        create_linear(4, 2, initializer="unknown")

    columns = get_test_data(
        16, sparse_feature_num=2, dense_feature_num=1, sequence_feature=[]
    )[2]
    autoint = AutoInt(columns, columns, device="cpu")
    assert autoint.dnn.linears[0].out_features == 256
    assert autoint.dnn.linears[1].out_features == 128
    assert autoint.dnn.linears[2].out_features == 64
    assert autoint.int_layers[0].W_Query.shape == (autoint.embedding_size, 16)
    assert autoint.int_layers[1].W_Query.shape == (16, 16)
    assert autoint.dnn_linear.in_features == 64 + len(autoint.embedding_dict) * 16

    difm = DIFM(columns, columns, device="cpu")
    assert difm.bit_wise_net.linears[0].out_features == 256
    assert difm.bit_wise_net.linears[1].out_features == 128
    assert difm.bit_wise_net.linears[2].out_features == 64
    assert difm.vector_wise_net.W_Query.shape == (difm.embedding_size, 64)
    assert difm.transform_matrix_P_vec.in_features == difm.sparse_feat_num * 64

    autoint = AutoInt(
        columns, columns, dnn_hidden_units=(8,), att_layer_num=1,
        l2_reg_linear=0.11, l2_reg_embedding=0.12, l2_reg_dnn=0.13,
        device="cpu",
    )
    registered = _registered_l2(autoint)
    assert registered["linear_model.embedding_dict.sparse_feature_0.weight"] == 0.11
    assert registered["embedding_dict.sparse_feature_0.weight"] == 0.12
    assert registered["dnn.linears.0.weight"] == 0.13
    assert "dnn_linear.weight" not in registered

    for model_type in (DeepFM, FiBiNET, NFM, ONN, WDL, xDeepFM):
        model = model_type(
            columns, columns, dnn_hidden_units=(8,),
            l2_reg_linear=0.11, l2_reg_embedding=0.12, l2_reg_dnn=0.13,
            device="cpu",
        )
        registered = _registered_l2(model)
        assert registered["dnn.linears.0.weight"] == 0.13
        assert "dnn_linear.weight" not in registered

    for model_type in (DCN, DCNMix):
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
            value for name, value in registered.items()
            if name.startswith("crossnet.")
        ]
        assert set(cross_values) == {0.14}

    ifm = IFM(columns, columns, dnn_hidden_units=(8,), l2_reg_dnn=0.13)
    registered = _registered_l2(ifm)
    assert registered["factor_estimating_net.linears.0.weight"] == 0.13
    assert "transform_weight_matrix_P.weight" not in registered

    difm = DIFM(columns, columns, dnn_hidden_units=(8,), l2_reg_dnn=0.13)
    registered = _registered_l2(difm)
    assert registered["bit_wise_net.linears.0.weight"] == 0.13
    assert not any(name.startswith("vector_wise_net.") for name in registered)
    assert "transform_matrix_P_vec.weight" not in registered
    assert "transform_matrix_P_bit.weight" not in registered

    _, _, din_columns, behavior_features = get_xy_fd()
    din = DIN(
        din_columns, behavior_features, dnn_hidden_units=(8,),
        l2_reg_embedding=0.12, l2_reg_dnn=0.13,
    )
    registered = _registered_l2(din)
    assert registered["dnn.linears.0.weight"] == 0.13
    assert "dnn_linear.weight" not in registered
