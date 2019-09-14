import torch.nn as nn
import torch.nn.functional as F
import torch

from .basemodel import *
from ..inputs import combined_dnn_input
from ..layers import DNN


class NFFM(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, embedding_size=4,
                 dnn_hidden_units=(128,128),
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5,l2_reg_dnn=0,
                 dnn_dropout=0, init_std=0.0001, seed=1024, dnn_use_bn=False, dnn_activation=F.relu,
                 task='binary', device='cpu'):
        """Instantiates the Wide&Deep Learning architecture.
        :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
        :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
        :param embedding_size: positive integer,sparse feature embedding_size
        :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
        :param l2_reg_linear: float. L2 regularizer strength applied to wide part
        :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
        :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
        :param init_std: float,to use as the initialize std of embedding vector
        :param seed: integer ,to use as random seed.
        :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
        :param dnn_activation: Activation function to use in DNN
        :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
        :param device:
        :return: A PyTorch model instance.
        """
        super(NFFM, self).__init__(linear_feature_columns, dnn_feature_columns, embedding_size=embedding_size,
                                  dnn_hidden_units=dnn_hidden_units,
                                  l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn, init_std=init_std,
                                  seed=seed,
                                  dnn_dropout=dnn_dropout, dnn_activation=dnn_activation,
                                  task=task, device=device)

        # first order part
        # has been done in the basemodel class

        # second order part
        self.second_order_embedding_dict = self.create_second_order_embedding_matrix(dnn_feature_columns, embedding_size=embedding_size, sparse=False)

        ## dnn part
        dim = self.__compute_nffm_dnn_dim(feature_columns=dnn_feature_columns, embedding_size=embedding_size)
        # print("dim: ", dim)
        self.dnn = DNN(inputs_dim=dim,
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn,
                       dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                       init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.add_regularization_loss(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2_reg_dnn)
        self.add_regularization_loss(self.dnn_linear.weight, l2_reg_dnn)
        self.to(device)

    def __compute_nffm_dnn_dim(self, feature_columns, embedding_size):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        return int(len(sparse_feature_columns) * embedding_size +\
                len(sparse_feature_columns)* (len(sparse_feature_columns) - 1) /2 * embedding_size  +\
               sum(map(lambda x: x.dimension, dense_feature_columns)))

    def forward(self, X):
        '''
        :param X: tensor (batch_size, feature_number)
        :return: y_pred tensor (batch_size, 1)
        '''

        # sparse_embedding_list: [(batch_size, 1, emb_size), ...]
        # dense_value_list: [(batch_size, 1), ...]
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        # second order  [(batch_size, 1,  emb_size), ...]
        spare_second_order_embedding_list = self.input_from_second_order_column(X, self.dnn_feature_columns,
                                                                                self.second_order_embedding_dict)
        # concat one order, second order and dense feature
        total_feature = []
        total_feature.extend(sparse_embedding_list)  # add one order
        total_feature.extend(spare_second_order_embedding_list)  # add second order
        total_feature = torch.cat(total_feature, dim=2).squeeze(1)  # (batch, n_f)
        total_feature = torch.cat([total_feature, torch.cat(dense_value_list, dim=1)], dim=1)  # add dense feature

        # print("total_feature_size: ", total_feature.size())
        dnn_output = self.dnn(total_feature)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)

        return y_pred


