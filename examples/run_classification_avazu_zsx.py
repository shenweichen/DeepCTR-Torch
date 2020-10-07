# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

if __name__ == "__main__":
    data = pd.read_csv('./avazu_sample.txt')
    data['day'] = data['hour'].apply(lambda x: str(x)[4:6])
    data['hour'] = data['hour'].apply(lambda x: str(x)[6:])
    print(data[:5])

    sparse_features = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
                       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
                       'device_model', 'device_type', 'device_conn_type',  # 'device_ip',
                       'C14',
                       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', ]
    print(len(sparse_features))  # 去掉id click device_ip   不用day  25-4=21

    data[sparse_features] = data[sparse_features].fillna('-1', )
    target = ['click']


    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2,random_state=2020)

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DCNMix(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary', low_rank=32, num_experts=4,
                   l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  # metrics=["binary_crossentropy", ], )
                  metrics=["binary_crossentropy", "auc"], )
    model.fit(train_model_input, train[target].values,
              batch_size=256, epochs=10, validation_split=0.0, verbose=2)

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
