from itertools import chain

import torch.nn as nn
import torch.nn.functional as F

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import DNN


class PNN(BaseModel):

    def __init__(self,
                 dnn_feature_columns, embedding_size=8, dnn_hidden_units=(128, 128),
                 l2_reg_embedding=1e-5, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', use_inner=True, use_outter=False,
                 kernel_type='mat', task='binary', device='cpu'):
        """Instantiates the DeepFM Network architecture.
        :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
        :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
        :param embedding_size: positive integer,sparse feature embedding_size
        :param use_fm: bool,use FM part or not
        :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
        :param l2_reg_linear: float. L2 regularizer strength applied to linear part
        :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
        :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
        :param init_std: float,to use as the initialize std of embedding vector
        :param seed: integer ,to use as random seed.
        :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
        :param dnn_activation: Activation function to use in DNN
        :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
        :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
        :param device:
        :return: A PyTorch model instance.
        """

        super(PNN, self).__init__(dnn_feature_columns, embedding_size=embedding_size,
                                  dnn_hidden_units=dnn_hidden_units,
                                  l2_reg_embedding=l2_reg_embedding, l2_reg_dnn=l2_reg_dnn,
                                  init_std=init_std, seed=seed,
                                  dnn_dropout=dnn_dropout, dnn_activation=dnn_activation,
                                  use_inner=use_inner, use_outter=use_outter, kernel_type=kernel_type,
                                  task=task, device=device)

        self.dnn = DNN(self.compute_input_dim(dnn_feature_columns, embedding_size, ), dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std, device=device)

        self.add_regularization_loss(
            filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2_reg_dnn)
        self.add_regularization_loss(self.dnn_linear.weight, l2_reg_dnn)


        self.to(device)

    def forward(self, X):

        if self.kernel_type not in ['mat', 'vec', 'num']:
            raise ValueError("kernel_type must be mat,vec or num")

        features = build_input_features(dnn_feature_columns)

        inputs_list = list(features.values())

        sparse_embedding_list, dense_value_list = input_from_feature_columns(features, slef.dnn_feature_columns,
                                                                                  self.embedding_size,
                                                                                  self.l2_reg_embedding,init_std,
                                                                                  self.seed)
        
        """
        inner_product = tf.keras.layers.Flatten()(InnerProductLayer()(sparse_embedding_list))
        outter_product = OutterProductLayer(kernel_type)(sparse_embedding_list)
                
            second order part (quadratic part)
        """
        if self.use_inner:
            inner_product_arr = []
            for i, weight_arr in enumerate(self.first_order_weight):
                tmp_arr = []
                for j, weight in enumerate(weight_arr):
                    tmp_arr.append(torch.sum(sparse_embedding_list[j] * weight, 1))
                sum_ = sum(tmp_arr)
                inner_product_arr.append((sum_*sum_).view([-1,1]))
            inner_product = torch.cat(inner_product_arr,1)
            first_order = sparse_embedding_list + inner_product

        if self.use_outer:
            outer_product_arr = []
            emb_arr_sum = sum(sparse_embedding_list)
            emb_matrix_arr = torch.bmm(emb_arr_sum.view([-1,self.embedding_size,1]),emb_arr_sum.view([-1,1,self.embedding_size]))
            for i, weight in enumerate(self.outer_second_weight_emb):
                outer_product_arr.append(torch.sum(torch.sum(emb_matrix_arr*weight,2),1).view([-1,1]))
            outer_product = torch.cat(outer_product_arr,1)
            first_order = sparse_embedding_list + outer_product

        dnn_output = self.dnn(first_order)
        

        y_pred = self.out(logit)

        return y_pred
