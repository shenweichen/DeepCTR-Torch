import torch
import torch.nn as nn
from .core import DNN

class KMaxPooling(nn.Module):
    """K Max pooling that selects the k biggest value along the specific axis.

      Input shape
        -  nD tensor with shape: ``(batch_size, ..., input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., output_dim)``.

      Arguments
        - **k**: positive integer, number of top elements to look for along the ``axis`` dimension.

        - **axis**: positive integer, the dimension to look for elements.

     """

    def __init__(self, k, axis, device='cpu'):
        super(KMaxPooling, self).__init__()
        self.k = k
        self.axis = axis
        self.to(device)

    def forward(self, input):
        if self.axis < 0 or self.axis >= len(input.shape):
            raise ValueError("axis must be 0~%d,now is %d" %
                             (len(input.shape)-1, self.axis))

        if self.k < 1 or self.k > input.shape[self.axis]:
            raise ValueError("k must be in 1 ~ %d,now k is %d" %
                             (input.shape[self.axis], self.k))

        out = torch.topk(input, k=self.k, dim=self.axis, sorted=True)[0]
        return out

class AttentionSequencePoolingLayer(nn.Module):
    """The Attentional sequence pooling operation used in DIN.

      Input shape
        - A list of three tensor: [query,keys,keys_length]

        - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``

        - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``

        - keys_mask is a 3D tensor with shape: ``(batch_size, 1, T)``

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **embedding_size**:positive integer, query features concat embedding size.
        - **att_hidden_units**:list of positive integer, the attention net layer number and units in each layer.
        - **att_activation**: Activation function to use in attention net.
        - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.
        - **return_score**:If True, return attention score.
        - **init_std**:float,to use as the initialize std of embedding vector
        - **use_bn**:bool. Whether use BatchNormalization before activation or not in deep net
        - **dropout_rate**:float in [0,1), the probability we will drop out a given DNN coordinate.
        - **l2_reg**:float. L2 regularizer strength applied to DNN
        - **seed**:integer ,to use as random seed.
        - **device**:str, ``"cpu"`` or ``"cuda:0"``

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, embedding_size, att_hidden_units=(80, 40), att_activation=F.relu, weight_normalization=False,
                 return_score=False, init_std=1e-4, use_bn=False, dropout_rate=0, l2_reg=0, seed=1024, device='cpu'):
        super(AttentionSequencePoolingLayer, self).__init__()
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        
        self.local_att = LocalActivationUnit(embedding_size, att_hidden_units, att_activation, l2_reg=l2_reg, dropout_rate=dropout_rate, 
                                             init_std=init_std, use_bn=use_bn, seed=seed, device=device)
        self.to(device)

    def forward(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 3:
            raise ValueError('A `AttentionSequencePoolingLayer` layer should be called '
                              'on a list of 3 inputs')

        if len(inputs[0].size()) != 3 or len(inputs[1].size()) != 3 or len(inputs[2].size()) != 3:
            raise ValueError(
                "Unexpected inputs dimensions,the 3 tensor dimensions are %d,%d and %d , expect to be 3,3 and 3" % (
                    len(inputs[0].size()), len(inputs[1].size()), len(inputs[2].size())))

        if inputs[0].size()[-1] != inputs[1].size()[-1] or inputs[0].size()[1] != 1 or inputs[2].size()[1] != 1:
            raise ValueError('A `AttentionSequencePoolingLayer` layer requires '
                              'inputs of a 3 tensor with shape (None,1,embedding_size),(None,T,embedding_size) and (None,1, T)'
                              'Got different shapes: %s' % (inputs))
        queries, keys, key_masks = inputs
        attention_score = self.local_att([queries, keys])

        outputs = attention_score.permute(0, 2, 1)

        if self.weight_normalization:
            paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        else:
            paddings = torch.zeros_like(outputs)

        outputs = torch.where(key_masks, outputs, paddings)

        if self.weight_normalization:
            outputs = nn.Softmax(outputs)

        if not self.return_score:
            outputs = torch.matmul(outputs, keys)

        return outputs

class LocalActivationUnit(nn.Module):
    """The LocalActivationUnit used in DIN with which the representation of
    user interests varies adaptively given different candidate items.

      Input shape
        - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``

      Output shape
        - 3D tensor with shape: ``(batch_size, T, 1)``.

      Arguments
        - **embedding_size**:positive integer, query features concat embedding size.
        - **hidden_units**:list of positive integer, the attention net layer number and units in each layer.
        - **activation**: Activation function to use in attention net.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.
        - **init_std**:float,to use as the initialize std of embedding vector
        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.
        - **seed**: A Python integer to use as random seed.
        - **device**:str, ``"cpu"`` or ``"cuda:0"``

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, embedding_size, hidden_units=(64, 32), activation=F.relu, l2_reg=0, dropout_rate=0, 
                 init_std=1e-4, use_bn=False, seed=1024, device='cpu'):
        super(LocalActivationUnit, self).__init__()
        self.dnn = DNN(4 * embedding_size, hidden_units,
                        activation=activation, l2_reg=l2_reg, dropout_rate=dropout_rate, use_bn=use_bn,
                        init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(hidden_units[-1], 1, bias=True).to(device)
        self.to(device)
    
    def forward(self, inputs):
        query, keys = inputs
        keys_len = keys.cpu().size(1)
        queries = torch.cat([query for _ in range(keys_len)], dim=1)
        attention_input = torch.cat([queries, keys, queries-keys, queries*keys], dim=-1)
        attention_output = self.dnn(attention_input)
        attention_output = self.dnn_linear(attention_output)
        return attention_output