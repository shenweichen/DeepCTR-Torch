# -*- coding: utf-8 -*-
import pytest
from deepctr_torch.models import AutoInt
from ..utils import get_test_data, SAMPLE_SIZE, check_model


@pytest.mark.parametrize(
    'att_layer_num,dnn_hidden_units,sparse_feature_num',
    [(1, (4, ), 2), (0, (4,), 2), (2, (4, 4,), 2), (1, (), 1), (1, (4,), 1)]
)
def test_AutoInt(att_layer_num, dnn_hidden_units, sparse_feature_num):
    # if version.parse(torch.__version__) >= version.parse("1.1.0") and len(dnn_hidden_units)==0:#todo check version
    #     return
    model_name = "AutoInt"
    sample_size = SAMPLE_SIZE
    x, y, feature_dim_dict = get_test_data(
        sample_size, sparse_feature_num, sparse_feature_num)

    model = AutoInt(feature_dim_dict, att_layer_num=att_layer_num,
                    dnn_hidden_units=dnn_hidden_units, dnn_dropout=0.5, )
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass