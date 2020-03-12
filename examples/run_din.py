import sys
sys.path.insert(0,'..')

import numpy as np
import torch
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models.din import DIN
import pdb

def get_xy_fd():
    # pdb.set_trace()
    feature_columns = [SparseFeat('user',3, embedding_dim=8),SparseFeat('gender', 2, embedding_dim=8), 
                       SparseFeat('item', 3 + 1, embedding_dim=8), SparseFeat('item_gender', 2 + 1, embedding_dim=8), 
                       DenseFeat('score', 1)]
    # feature_columns += [VarLenSparseFeat('hist_item', 3 + 1, maxlen=4, embedding_name='item'),
    #                     VarLenSparseFeat('hist_item_gender', 2 + 1, maxlen=4, embedding_name='item_gender')]
    
    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item', 3 + 1, embedding_dim=8), 4),
                        VarLenSparseFeat(SparseFeat('hist_item_gender', 2 + 1, embedding_dim=8), 4)]

    

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
    y = np.array([1, 0, 1])
    
    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    model = DIN(feature_columns, behavior_feature_list)
    model.compile('adagrad', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, batch_size=3, epochs=10, validation_split=0.0, verbose=2)
