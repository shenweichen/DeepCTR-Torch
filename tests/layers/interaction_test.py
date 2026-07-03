import torch
import torch.nn.functional as F

from deepctr_torch.layers.interaction import CIN, CrossNetMix


def test_cin_uses_tf_channel_order():
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


def test_cin_uses_glorot_uniform_and_zero_bias():
    layer = CIN(field_size=3, layer_size=(4, 3), split_half=False)
    for conv in layer.conv1ds:
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(conv.weight)
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        assert torch.max(torch.abs(conv.weight)) <= bound
        torch.testing.assert_close(conv.bias, torch.zeros_like(conv.bias))


def test_crossnetmix_preserves_batch_dimension_for_single_sample():
    layer = CrossNetMix(in_features=4, low_rank=2, num_experts=2, layer_num=2)
    output = layer(torch.randn(1, 4))
    assert output.shape == (1, 4)


def test_crossnetmix_gates_use_glorot_uniform():
    layer = CrossNetMix(in_features=4, low_rank=2, num_experts=2, layer_num=1)
    for gate in layer.gating:
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(gate.weight)
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        assert torch.max(torch.abs(gate.weight)) <= bound
        assert gate.bias is None
