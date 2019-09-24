# -*- coding: utf-8 -*-
import pytest
import sys
from deepctr_torch.models import CCPM
from tests.utils import get_test_data, SAMPLE_SIZE, check_model


@pytest.mark.parametrize(
    'sparse_feature_num,dense_feature_num',
    [(2,0),(3,0)
     ]
)
def test_CCPM(sparse_feature_num, dense_feature_num):
    model_name = "CCPM"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num, dense_feature_num)

    model = CCPM(feature_columns, feature_columns,
                dnn_hidden_units=[32, ], dnn_dropout=0.5, conv_kernel_width=(6, 5),
                 conv_filters=(4, 4))
    check_model(model, model_name, x, y,check_model_io=False)


if __name__ == "__main__":
   pass