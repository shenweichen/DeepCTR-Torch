# -*- coding: utf-8 -*-
import pytest

from deepctr_torch.models import AFN
from tests.utils import get_test_data, SAMPLE_SIZE, check_model, get_device


@pytest.mark.parametrize(
    'afn_dnn_hidden_units, sparse_feature_num, dense_feature_num',
    [((256, 128), 3, 0),
     ((256, 128), 3, 3),
     ((256, 128), 0, 3)]
)
def test_AFN(afn_dnn_hidden_units, sparse_feature_num, dense_feature_num):
    model_name = 'AFN'
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=dense_feature_num)

    model = AFN(feature_columns, feature_columns, afn_dnn_hidden_units=afn_dnn_hidden_units, device=get_device())

    check_model(model, model_name, x, y)


if __name__ == '__main__':
    pass
