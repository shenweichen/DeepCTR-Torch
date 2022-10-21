# Quick-Start

## Installation Guide
`deepctr-torch` depends on torch>=1.2.0, you can specify to install it through `pip`.

```bash
$ pip install -U deepctr-torch
```
## Getting started: 4 steps to DeepCTR-Torch


### Step 1: Import model


```python
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names

data = pd.read_csv('./criteo_sample.txt')

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
target = ['label']
```
    


### Step 2: Simple preprocessing


Usually there are two simple way to encode the sparse categorical feature for embedding

- Label Encoding: map the features to integer value from 0 ~ len(#unique) - 1
  ```python
  for feat in sparse_features:
      lbe = LabelEncoder()
      data[feat] = lbe.fit_transform(data[feat])
  ```
- Hash Encoding: 【Currently not supported】.

And for dense numerical features,they are usually  discretized to buckets,here we use normalization.

```python
mms = MinMaxScaler(feature_range=(0,1))
data[dense_features] = mms.fit_transform(data[dense_features])
```


### Step 3: Generate feature columns

For sparse features, we transform them into dense vectors by embedding techniques.
For dense numerical features, we concatenate them to the input tensors of fully connected layer.

- Label Encoding
```python
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4)
                       for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                      for feat in dense_features]
```
- Feature Hashing on the fly【currently not supported】
```python
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=1e6,embedding_dim=4, use_hash=True, dtype='string')  # since the input is string
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                          for feat in dense_features]
```
- generate feature columns
```python
dnn_feature_columns = sparse_feature_columns + dense_feature_columns
linear_feature_columns = sparse_feature_columns + dense_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

```
### Step 4: Generate the training samples and train the model

```python
train, test = train_test_split(data, test_size=0.2)
train_model_input = {name:train[name] for name in feature_names}

test_model_input = {name:test[name] for name in feature_names}


device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

model = DeepFM(linear_feature_columns,dnn_feature_columns,task='binary',device=device)
model.compile("adam", "binary_crossentropy",
              metrics=['binary_crossentropy'], )

history = model.fit(train_model_input,train[target].values,batch_size=256,epochs=10,verbose=2,validation_split=0.2)
pred_ans = model.predict(test_model_input, batch_size=256)

```
You can check the full code [here](./Examples.html#classification-criteo).








