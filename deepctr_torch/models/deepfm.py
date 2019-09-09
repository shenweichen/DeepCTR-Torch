import torch
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import FM, DNN


class DeepFM(BaseModel):

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, embedding_size=8, use_fm=True,
                 dnn_hidden_units=(128, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation=F.relu, dnn_use_bn=False, task='binary', device='cpu'):

        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, embedding_size=embedding_size,
                                     dnn_hidden_units=dnn_hidden_units,
                                     l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn, init_std=init_std,
                                     seed=seed,
                                     dnn_dropout=dnn_dropout, dnn_activation=dnn_activation,
                                     task=task, device=device)

        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns, embedding_size, ), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,use_bn=dnn_use_bn, init_std=init_std)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)

        self.add_regularization_loss(self.dnn.weight, l2_reg_dnn)
        self.add_regularization_loss(self.dnn_linear.weight,l2_reg_dnn)

        if use_fm:
            self.fm = FM()
        self.use_fm = use_fm
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        linear_logit = self.linear_model(X)

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        logit = linear_logit + dnn_logit

        if self.use_fm:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)
        y_pred = self.out(logit)

        return y_pred
