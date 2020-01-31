# -*- coding: utf-8 -*-
import pytest

from deepctr_torch.models import AFM
from ..utils import get_test_data, SAMPLE_SIZE, check_model, get_device


@pytest.mark.parametrize(
    'use_attention, sparse_feature_num, dense_feature_num',
    [(True, 3, 0), ]
)
def test_AFM(use_attention, sparse_feature_num, dense_feature_num):
    model_name = 'AFM'
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=dense_feature_num)

    model = AFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                use_attention=use_attention, afm_dropout=0.5, device=get_device())

    check_model(model, model_name, x, y)


if __name__ == '__main__':
    pass
