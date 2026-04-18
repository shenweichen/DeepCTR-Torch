# -*- coding: utf-8 -*-
import pytest

from deepctr_torch.models import DeepFM
from ..utils import get_test_data, SAMPLE_SIZE, check_model, get_device


@pytest.mark.parametrize(
    'use_fm,hidden_size,sparse_feature_num,dense_feature_num',
    [(True, (32,), 3, 3),
     (False, (32,), 3, 3),
     (False, (32,), 2, 2),
     (False, (32,), 1, 1),
     (True, (), 1, 1),
     (False, (), 2, 2),
     (True, (32,), 0, 3),
     (True, (32,), 3, 0),
     (False, (32,), 0, 3),
     (False, (32,), 3, 0),
     ]
)
def test_DeepFM(use_fm, hidden_size, sparse_feature_num, dense_feature_num):
    model_name = "DeepFM"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=dense_feature_num)

    model = DeepFM(feature_columns, feature_columns, use_fm=use_fm,
                   dnn_hidden_units=hidden_size, dnn_dropout=0.5, device=get_device())
    check_model(model, model_name, x, y)

    # no linear part
    model = DeepFM([], feature_columns, use_fm=use_fm,
                   dnn_hidden_units=hidden_size, dnn_dropout=0.5, device=get_device())
    check_model(model, model_name + '_no_linear', x, y)


def test_DeepFM_fit_with_column_vector_target():
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num=2, dense_feature_num=2)

    model = DeepFM(feature_columns, feature_columns, dnn_hidden_units=(8,), dnn_dropout=0.5, device=get_device())
    model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy'])

    history = model.fit(x, y.reshape(-1, 1), batch_size=32, epochs=1, verbose=0, validation_split=0.2)
    assert "loss" in history.history


if __name__ == "__main__":
    pass
