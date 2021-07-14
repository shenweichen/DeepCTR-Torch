"""
Date: 2021-07-13 11:59:44
LastEditors: GodK
LastEditTime: 2021-07-13 12:26:11
"""
import pytest
from deepctr_torch.models import DeepFM
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from .utils import get_test_data, SAMPLE_SIZE,get_device

@pytest.mark.parametrize(
    'use_fm,hidden_size,sparse_feature_num',
    [(True, (32,), 3),
     (False, (32,), 3),
     (False, (32,), 2), (False, (32,), 1), (True, (), 1), (False, (), 2)
     ]
)
def test_save(use_fm, hidden_size, sparse_feature_num):
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=sparse_feature_num)
    model = DeepFM(feature_columns, feature_columns, use_fm=use_fm,
                   dnn_hidden_units=hidden_size, dnn_dropout=0.5, device=get_device())
    
    # test callbacks
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, verbose=1, patience=0, mode='max')
    model_checkpoint = ModelCheckpoint(filepath='model.ckpt', monitor='val_acc', verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False, mode='max', period=1)
    model.compile('adam', 'binary_crossentropy', metrics=['binary_crossentropy','acc'])
    model.fit(x, y, batch_size=64, epochs=3, validation_split=0.5, callbacks=[early_stopping, model_checkpoint])

if __name__ == "__main__":
    pass