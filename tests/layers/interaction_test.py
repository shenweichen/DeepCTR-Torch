import torch
import torch.nn.functional as F
from unittest.mock import patch

from deepctr_torch.layers.interaction import (
    BilinearInteraction,
    CIN,
    CrossNetMix,
    InteractingLayer,
    SENETLayer,
)


def test_cin_matches_tf_channel_order_and_initialization():
    layer = CIN(field_size=2, layer_size=(2, 2), split_half=False,
                activation='linear')
    with torch.no_grad():
        layer.conv1ds[0].weight.copy_(torch.arange(8).reshape(2, 4, 1))
        layer.conv1ds[0].bias.zero_()
        layer.conv1ds[1].weight.copy_(torch.arange(8).reshape(2, 4, 1) / 10)
        layer.conv1ds[1].bias.zero_()

    inputs = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]])
    hidden = inputs
    expected_outputs = []
    for conv in layer.conv1ds:
        interactions = torch.einsum('bmd,bhd->bmhd', inputs, hidden)
        interactions = interactions.reshape(1, inputs.shape[1] * hidden.shape[1], inputs.shape[2])
        hidden = F.conv1d(interactions, conv.weight, conv.bias)
        expected_outputs.append(hidden)
    expected = torch.cat(expected_outputs, dim=1).sum(dim=-1)

    torch.testing.assert_close(layer(inputs), expected)

    initialized_layer = CIN(
        field_size=3, layer_size=(4, 3), split_half=False)
    for conv in initialized_layer.conv1ds:
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(conv.weight)
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        assert torch.max(torch.abs(conv.weight)) <= bound
        torch.testing.assert_close(conv.bias, torch.zeros_like(conv.bias))


def test_other_interaction_layers_match_tf():
    layer = CrossNetMix(in_features=4, low_rank=2, num_experts=2, layer_num=2)
    output = layer(torch.randn(1, 4))
    assert output.shape == (1, 4)

    layer = CrossNetMix(in_features=4, low_rank=2, num_experts=2, layer_num=1)
    for gate in layer.gating:
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(gate.weight)
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        assert torch.max(torch.abs(gate.weight)) <= bound
        assert gate.bias is None

    with patch('torch.nn.init.xavier_normal_', wraps=torch.nn.init.xavier_normal_) as initializer:
        layer = SENETLayer(filed_size=6, reduction_ratio=3)
    assert initializer.call_count == 2
    assert layer.excitation[0].bias is None
    assert layer.excitation[2].bias is None

    layer = BilinearInteraction(filed_size=4, embedding_size=3,
                                bilinear_type='each')
    assert len(layer.bilinear) == 3
    assert layer.bilinear[0].weight.numel() == 9
    assert layer.bilinear[1].weight.numel() == 9
    assert layer.bilinear[2].weight.numel() == 9

    with patch('torch.nn.init.xavier_normal_', wraps=torch.nn.init.xavier_normal_) as initializer:
        layer = BilinearInteraction(filed_size=3, embedding_size=4,
                                    bilinear_type='interaction')
    assert initializer.call_count == 3
    assert layer.bilinear[0].bias is None
    assert layer.bilinear[1].bias is None
    assert layer.bilinear[2].bias is None

    layer = InteractingLayer(embedding_size=8, head_num=2, use_res=True)
    for parameter in layer.parameters():
        assert torch.max(torch.abs(parameter)) <= 0.1

    layer = InteractingLayer(
        embedding_size=6, head_num=2, att_embedding_size=4, use_res=True)
    output = layer(torch.randn(3, 5, 6))
    assert output.shape == (3, 5, 8)
    assert layer.W_Query.shape == (6, 8)
    assert layer.W_Res.shape == (6, 8)
