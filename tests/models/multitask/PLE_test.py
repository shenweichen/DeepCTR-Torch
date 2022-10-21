# -*- coding: utf-8 -*-
import pytest

from deepctr_torch.models import PLE
from ...utils_mtl import get_mtl_test_data, SAMPLE_SIZE, check_mtl_model, get_device


@pytest.mark.parametrize(
    'shared_expert_num, specific_expert_num, num_levels, expert_dnn_hidden_units, gate_dnn_hidden_units, '
    'tower_dnn_hidden_units, task_types, sparse_feature_num ,dense_feature_num',
    [
        (1, 1, 2, (32, 16), (64,), (64,), ['binary', 'binary'], 3, 3),
        (3, 3, 3, (32, 16), (), (64,), ['binary', 'binary'], 3, 3),
        (3, 3, 3, (32, 16), (64,), (), ['binary', 'binary'], 3, 3),
        (3, 3, 3, (32, 16), (), (), ['binary', 'binary'], 3, 3),
        (3, 3, 3, (32, 16), (64,), (64,), ['binary', 'regression'], 3, 3),
    ]
)
def test_PLE(shared_expert_num, specific_expert_num, num_levels, expert_dnn_hidden_units, gate_dnn_hidden_units,
             tower_dnn_hidden_units, task_types, sparse_feature_num, dense_feature_num):
    model_name = "PLE"
    sample_size = SAMPLE_SIZE
    x, y_list, feature_columns = get_mtl_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=dense_feature_num)

    model = PLE(feature_columns, shared_expert_num=shared_expert_num, specific_expert_num=specific_expert_num,
                num_levels=num_levels, expert_dnn_hidden_units=expert_dnn_hidden_units,
                gate_dnn_hidden_units=gate_dnn_hidden_units, tower_dnn_hidden_units=tower_dnn_hidden_units,
                task_types=task_types, device=get_device())
    check_mtl_model(model, model_name, x, y_list, task_types)


if __name__ == "__main__":
    pass
