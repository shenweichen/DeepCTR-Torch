# FAQ

## 1. Save or load weights/models
----------------------------------------
To save/load weights:

```python
import torch
model = DeepFM()
torch.save(model.state_dict(), 'DeepFM_weights.h5')
model.load_state_dict(torch.load('DeepFM_weights.h5'))
```

To save/load models:

```python
import torch
model = DeepFM()
torch.save(model, 'DeepFM.h5')
model = torch.load('DeepFM.h5')
```

## 2. How to add a long dense feature vector as a input to the model?
```python
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import DenseFeat,SparseFeat,get_feature_names
import numpy as np

feature_columns = [SparseFeat('user_id',120,),SparseFeat('item_id',60,),DenseFeat("pic_vec",5)]
fixlen_feature_names = get_feature_names(feature_columns)

user_id = np.array([[1],[0],[1]])
item_id = np.array([[30],[20],[10]])
pic_vec = np.array([[0.1,0.5,0.4,0.3,0.2],[0.1,0.5,0.4,0.3,0.2],[0.1,0.5,0.4,0.3,0.2]])
label = np.array([1,0,1])

model_input = {'user_id':user_id,'item_id':item_id,'pic_vec':pic_vec}

model = DeepFM(feature_columns,feature_columns)
model.compile('adagrad','binary_crossentropy')
model.fit(model_input,label)
```

## 3. How to run the demo with GPU ?
```python
import torch
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

model = DeepFM(...,device=device)
```
