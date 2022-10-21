# Examples


## Classification: Criteo 

The Criteo Display Ads dataset is for the purpose of predicting ads 
click-through rate. It has 13 integer features and
26 categorical features where each category has a high cardinality.

![image](../pics/criteo_sample.png)

In this example,we simply normailize the dense feature between 0 and 1,you
can try other transformation technique like log normalization or discretization.Then we use [SparseFeat](./Features.html#sparsefeat) and [DenseFeat](./Features.html#densefeat) to generate feature columns  for sparse features and dense features.


This example shows how to use ``DeepFM`` to solve a simple binary classification task. You can get the demo data [criteo_sample.txt](https://github.com/shenweichen/DeepCTR-Torch/tree/master/examples/criteo_sample.txt)
and run the following codes.

```python
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')

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
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )
    model.fit(train_model_input,train[target].values,batch_size=32,epochs=10,verbose=2,validation_split=0.0)

    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

```

## Regression: Movielens

The MovieLens data has been used for personalized tag recommendation,which
contains 668, 953 tag applications of users on movies.
Here is a small fraction of data include  only sparse field.

![image](../pics/movielens_sample.png)


This example shows how to use ``DeepFM`` to solve a simple binary regression task. You can get the demo data 
[movielens_sample.txt](https://github.com/shenweichen/DeepCTR-Torch/tree/master/examples/movielens_sample.txt) and run the following codes.

```python
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM

if __name__ == "__main__":

    data = pd.read_csv("./movielens_sample.txt")
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip"]
    target = ['rating']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # 2.count #unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=device)
    model.compile("adam", "mse", metrics=['mse'], )

    history = model.fit(train_model_input,train[target].values,batch_size=256,epochs=10,verbose=2,validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test MSE", round(mean_squared_error(
        test[target].values, pred_ans), 4))


```

## Multi-value Input : Movielens
----------------------------------

The MovieLens data has been used for personalized tag recommendation,which
contains 668, 953 tag applications of users on movies.
Here is a small fraction of data include  sparse fields and a multivalent field.

![image](../pics/movielens_sample_with_genres.png)

There are 2 additional steps to use DeepCTR with sequence feature input.

1. Generate the paded and encoded sequence feature  of sequence input feature(**value 0 is for padding**).
2. Generate config of sequence feature with [VarLenSparseFeat](./Features.html#varlensparsefeat)


This example shows how to use ``DeepFM`` with sequence(multi-value) feature. You can get the demo data 
[movielens_sample.txt](https://github.com/shenweichen/DeepCTR-Torch/tree/master/examples/movielens_sample.txt) and run the following codes.

```python
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr_torch.models import DeepFM


def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


if __name__ == "__main__":
    data = pd.read_csv("./movielens_sample.txt")
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip", ]
    target = ['rating']

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # preprocess the sequence feature

    key2index = {}
    genres_list = list(map(split, data['genres'].values))
    genres_length = np.array(list(map(len, genres_list)))
    max_len = max(genres_length)
    # Notice : padding=`post`
    genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                              for feat in sparse_features]

    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
        key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean')]  # Notice : value 0 is for padding for sequence input feature

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    model_input = {name: data[name] for name in sparse_features}  #
    model_input["genres"] = genres_list

    # 4.Define Model,compile and train

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=device)

    model.compile("adam", "mse", metrics=['mse'], )
    history = model.fit(model_input,data[target].values,batch_size=256,epochs=10,verbose=2,validation_split=0.2)


```

## MultiTask Learning:MMOE

This example shows how to use ``MMOE`` to solve a multi task learning problem. You can get the demo
data [byterec_sample.txt](https://github.com/shenweichen/DeepCTR-Torch/tree/master/examples/byterec_sample.txt) and run
the following codes.

```python
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
```