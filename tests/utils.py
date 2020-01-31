# -*- coding: utf-8 -*-
import os

import numpy as np
import torch as torch

from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat

SAMPLE_SIZE = 64


def gen_sequence(dim, max_len, sample_size):
    return np.array([np.random.randint(0, dim, max_len) for _ in range(sample_size)]), np.random.randint(1, max_len + 1,
                                                                                                         sample_size)


def get_test_data(sample_size=1000, embedding_size=4, sparse_feature_num=1, dense_feature_num=1,
                  sequence_feature=['sum', 'mean', 'max'], classification=True, include_length=False,
                  hash_flag=False, prefix=''):


    feature_columns = []
    model_input = {}


    if 'weight'  in sequence_feature:
        feature_columns.append(VarLenSparseFeat(SparseFeat(prefix+"weighted_seq",vocabulary_size=2,embedding_dim=embedding_size),maxlen=3,length_name=prefix+"weighted_seq"+"_seq_length",weight_name=prefix+"weight"))
        s_input, s_len_input = gen_sequence(
            2, 3, sample_size)

        model_input[prefix+"weighted_seq"] = s_input
        model_input[prefix+'weight'] = np.random.randn(sample_size,3,1)
        model_input[prefix+"weighted_seq"+"_seq_length"] = s_len_input
        sequence_feature.pop(sequence_feature.index('weight'))


    for i in range(sparse_feature_num):
        dim = np.random.randint(1, 10)
        feature_columns.append(SparseFeat(prefix+'sparse_feature_'+str(i), dim,embedding_size,dtype=torch.int32))
    for i in range(dense_feature_num):
        feature_columns.append(DenseFeat(prefix+'dense_feature_'+str(i), 1,dtype=torch.float32))
    for i, mode in enumerate(sequence_feature):
        dim = np.random.randint(1, 10)
        maxlen = np.random.randint(1, 10)
        feature_columns.append(
            VarLenSparseFeat(SparseFeat(prefix +'sequence_' + mode,vocabulary_size=dim,  embedding_dim=embedding_size), maxlen=maxlen, combiner=mode))

    for fc in feature_columns:
        if isinstance(fc,SparseFeat):
            model_input[fc.name]= np.random.randint(0, fc.vocabulary_size, sample_size)
        elif isinstance(fc,DenseFeat):
            model_input[fc.name] = np.random.random(sample_size)
        else:
            s_input, s_len_input = gen_sequence(
                fc.vocabulary_size, fc.maxlen, sample_size)
            model_input[fc.name] = s_input
            if include_length:
                fc.length_name = prefix+"sequence_"+str(i)+'_seq_length'
                model_input[prefix+"sequence_"+str(i)+'_seq_length'] = s_len_input

    if classification:
        y = np.random.randint(0, 2, sample_size)
    else:
        y = np.random.random(sample_size)

    return model_input, y, feature_columns


def check_model(model, model_name, x, y, check_model_io=True):
    '''
    compile model,train and evaluate it,then save/load weight and model file.
    :param model:
    :param model_name:
    :param x:
    :param y:
    :param check_model_io:
    :return:
    '''

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    model.fit(x, y, batch_size=100, epochs=1, validation_split=0.5)

    print(model_name + 'test, train valid pass!')
    torch.save(model.state_dict(), model_name + '_weights.h5')
    model.load_state_dict(torch.load(model_name + '_weights.h5'))
    os.remove(model_name + '_weights.h5')
    print(model_name + 'test save load weight pass!')
    if check_model_io:
        torch.save(model, model_name + '.h5')
        model = torch.load(model_name + '.h5')
        os.remove(model_name + '.h5')
        print(model_name + 'test save load model pass!')
    print(model_name + 'test pass!')
