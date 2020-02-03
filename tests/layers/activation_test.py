# -*- coding: utf-8 -*-
from deepctr_torch.layers import activation
from tests.utils import layer_test


def test_dice():
    layer_test(activation.Dice, kwargs={'num_features': 3, 'dim': 2},
               input_shape=(5, 3), expected_output_shape=(5,3))
