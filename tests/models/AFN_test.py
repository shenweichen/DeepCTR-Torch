# -*- coding: utf-8 -*-
import pytest

from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from deepctr_torch.models import AFN
from tests.utils import get_test_data, SAMPLE_SIZE, check_model, get_device


@pytest.mark.parametrize(
    'dnn_hidden_units, sparse_feature_num, dense_feature_num',
    [((256, 128), 3, 0), 
    ((256, 128), 3, 3),
     ((256, 128), 0,3),
     ((), 3,0),
     ((), 3,3),
     ((), 0,0)]
)

def test_AFN(dnn_hidden_units, sparse_feature_num, dense_feature_num):
    model_name = 'AFN'
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=dense_feature_num)

    model = AFN(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                dnn_hidden_units=dnn_hidden_units, device=get_device())

    check_model(model, model_name, x, y)

if __name__ == '__main__':
    pass
