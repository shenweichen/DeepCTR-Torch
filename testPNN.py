import pandas as pd
import numpy as np
from deepctr_torch.models import PNN
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat,get_fixlen_feature_names
# InnerProductLayer, OutterProductLayer

import torch


## data preprocess

data = pd.read_csv('examples/criteo_sample.txt')

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
target = ['label']

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

# 2.count #unique features for each sparse field,and record dense feature field name

fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                          for feat in sparse_features] + [DenseFeat(feat, 1,)
                                                          for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

fixlen_feature_names = get_fixlen_feature_names(
    linear_feature_columns + dnn_feature_columns)

# 3.generate input data for model

train, test = train_test_split(data, test_size=0.2)
train_model_input = [train[name] for name in fixlen_feature_names]
test_model_input = [test[name] for name in fixlen_feature_names]

device = 'cpu'
use_cuda = False
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

model = PNN(linear_feature_columns, dnn_feature_columns, use_inner=False, use_outter=True,
                     device=device, task="binary")

model.compile("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"],)
model.fit(train_model_input, train[target].values, batch_size=32, epochs=1, validation_split=0.2, verbose=2)

pred_ans = model.predict(test_model_input, 32)
#print(pred_ans)
print("")
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))