import torch
import torch.nn as nn


class SequencePoolingLayer(nn.Module):
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.

      Input shape
        - A list of two  tensor [seq_value,seq_len]

        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``

        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **mode**:str.Pooling operation to be used,can be sum,mean or max.

    """

    def __init__(self, mode='mean', supports_masking=False, device='cpu'):

        super(SequencePoolingLayer, self).__init__()
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError('parameter mode should in [sum, mean, max]')
        self.supports_masking = supports_masking
        self.mode = mode
        self.eps = torch.FloatTensor([1e-8]).to(device)
        self.to(device)


    def _sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        # Returns a mask tensor representing the first N positions of each cell.
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix

        mask.type(dtype)
        return mask

    def forward(self, seq_value_len_list):
        if self.supports_masking:
            uiseq_embed_list, mask = seq_value_len_list  # [B, T, E], [B, 1]
            mask = mask.float()
            user_behavior_length = torch.sum(mask, dim=-1, keepdim=True)
            mask = mask.unsqueeze(2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list  # [B, T, E], [B, 1]
            mask = self._sequence_mask(user_behavior_length, maxlen=uiseq_embed_list.shape[1],
                                       dtype=torch.float32)  # [B, 1, maxlen]
            mask = torch.transpose(mask, 1, 2)  # [B, maxlen, 1]

        embedding_size = uiseq_embed_list.shape[-1]

        mask = torch.repeat_interleave(mask, embedding_size, dim=2)  # [B, maxlen, E]

        if self.mode == 'max':
            hist = uiseq_embed_list - (1 - mask) * 1e9
            hist = torch.max(hist, dim=1, keepdim=True)[0]
            return hist
        hist = torch.sum(uiseq_embed_list * mask, dim=1, keepdim=False)

        if self.mode == 'mean':
            hist = torch.div(hist, user_behavior_length.type(torch.float32) + self.eps)

        hist = torch.unsqueeze(hist, dim=1)
        return hist


# class AttentionSequencePoolingLayer(nn.Module):
#     """The Attentional sequence pooling operation used in DIN.
#
#       Input shape
#         - A list of three tensor: [query,keys,keys_length]
#
#         - query is a 3D tensor with shape:  ``(batch_size, 1, embedding_size)``
#
#         - keys is a 3D tensor with shape:   ``(batch_size, T, embedding_size)``
#
#         - keys_length is a 2D tensor with shape: ``(batch_size, 1)``
#
#       Output shape
#         - 3D tensor with shape: ``(batch_size, 1, embedding_size)``
#
#       Arguments
#         - **att_hidden_units**: List of positive integer, the attention net layer number and units in each layer.
#
#         - **embedding_dim**: Dimension of the input embeddings.
#
#         - **activation**: Activation function to use in attention net.
#
#         - **weight_normalization**: bool.Whether normalize the attention score of local activation unit.
#
#       References
#         - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
#     """
#
#     def __init__(self, att_hidden_units=[80, 40], embedding_dim=4, activation='Dice', weight_normalization=False):
#         super(AttentionSequencePoolingLayer, self).__init__()
#
#         self.local_att = LocalActivationUnit(hidden_units=att_hidden_units, embedding_dim=embedding_dim,
#                                              activation=activation)
#
#     def forward(self, query, keys, keys_length):
#         # query: [B, 1, E], keys: [B, T, E], keys_length: [B, 1]
#         # TODO: Mini-batch aware regularization in originial paper [Zhou G, et al. 2018] is not implemented here. As the authors mentioned
#         #       it is not a must for small dataset as the open-sourced ones.
#         attention_score = self.local_att(query, keys)
#         attention_score = torch.transpose(attention_score, 1, 2)  # B * 1 * T
#
#         # define mask by length
#         keys_length = keys_length.type(torch.LongTensor)
#         mask = torch.arange(keys.size(1))[None, :] < keys_length[:, None]  # [1, T] < [B, 1, 1]  -> [B, 1, T]
#
#         # mask
#         output = torch.mul(attention_score, mask.type(torch.FloatTensor))  # [B, 1, T]
#
#         # multiply weight
#         output = torch.matmul(output, keys)  # [B, 1, E]
#
#         return output


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
                             (len(input.shape) - 1, self.axis))

        if self.k < 1 or self.k > input.shape[self.axis]:
            raise ValueError("k must be in 1 ~ %d,now k is %d" %
                             (input.shape[self.axis], self.k))

        out = torch.topk(input, k=self.k, dim=self.axis, sorted=True)[0]
        return out
