# -*- coding: utf-8 -*-
import pytest

from deepctr_torch.models import ESMM
from ...utils_mtl import get_mtl_test_data, SAMPLE_SIZE, check_mtl_model, get_device


@pytest.mark.parametrize(
    'num_experts, tower_dnn_hidden_units, task_types, sparse_feature_num, dense_feature_num',
    [
        (3, (32, 16), ['binary', 'binary'], 3, 3)
    ]
)
def test_ESMM(num_experts, tower_dnn_hidden_units, task_types,
              sparse_feature_num, dense_feature_num):
    model_name = "ESMM"
    sample_size = SAMPLE_SIZE
    x, y_list, feature_columns = get_mtl_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=dense_feature_num)

    model = ESMM(feature_columns, tower_dnn_hidden_units=tower_dnn_hidden_units,
                 task_types=task_types, device=get_device())
    check_mtl_model(model, model_name, x, y_list, task_types)


if __name__ == "__main__":
    pass
