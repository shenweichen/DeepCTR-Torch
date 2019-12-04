# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.models import *
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
import torch
import torch.nn.functional as F

feature_columns = [SparseFeat('user',3),SparseFeat(
        'gender', 2), SparseFeat('item', 3 + 1), SparseFeat('item_gender', 2 + 1),DenseFeat('score', 1)]
feature_columns += [VarLenSparseFeat('hist_item',3+1, maxlen=4, embedding_name='item'),
                    VarLenSparseFeat('hist_item_gender',3+1, maxlen=4, embedding_name='item_gender')]

behavior_feature_list = ["item", "item_gender"]
uid = np.array([0, 1, 2])
ugender = np.array([0, 1, 0])
iid = np.array([1, 2, 3])  # 0 is mask value
igender = np.array([1, 2, 1])  # 0 is mask value
score = np.array([0.1, 0.2, 0.3])

hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])

feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                'hist_item': hist_iid, 'hist_item_gender': hist_igender, 'score': score}
x = {name:feature_dict[name] for name in get_feature_names(feature_columns)}
y = np.array([[1], [0], [1]])

model = DIN([], feature_columns, behavior_feature_list, hist_len_max=4, device='cpu', dnn_activation='dice', att_activation='dice')
model.compile('adagrad', 'binary_crossentropy', metrics=['binary_crossentropy'])
model.fit(x, y, batch_size=32, epochs=3, validation_split=0.0, verbose=1)