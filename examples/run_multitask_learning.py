# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

if __name__ == "__main__":
    # data description can be found in https://www.biendata.xyz/competition/icmechallenge2019/
    data = pd.read_csv('./byterec_sample.txt', sep='\t',
                       names=["uid", "user_city", "item_id", "author_id", "item_city", "channel", "finish", "like",
                              "music_id", "device", "time", "duration_time"])

    sparse_features = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "music_id", "device"]
    dense_features = ["duration_time"]

    target = ['finish', 'like']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    split_boundary = int(data.shape[0] * 0.8)
    train, test = data[:split_boundary], data[split_boundary:]
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = MMOE(dnn_feature_columns, task_types=['binary', 'binary'],
                 l2_reg_embedding=1e-5, task_names=target, device=device)
    model.compile("adagrad", loss=["binary_crossentropy", "binary_crossentropy"],
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values, batch_size=32, epochs=10, verbose=2)
    pred_ans = model.predict(test_model_input, 256)
    print("")
    for i, target_name in enumerate(target):
        print("%s test LogLoss" % target_name, round(log_loss(test[target[i]].values, pred_ans[:, i]), 4))
        print("%s test AUC" % target_name, round(roc_auc_score(test[target[i]].values, pred_ans[:, i]), 4))
