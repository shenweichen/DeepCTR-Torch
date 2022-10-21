# -*- coding: utf-8 -*-
import os

import numpy as np
import torch as torch

from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat

SAMPLE_SIZE = 64


def gen_sequence(dim, max_len, sample_size):
    return np.array([np.random.randint(0, dim, max_len) for _ in range(sample_size)]), np.random.randint(1, max_len + 1,
                                                                                                         sample_size)


def get_mtl_test_data(sample_size=1000, embedding_size=4, sparse_feature_num=1, dense_feature_num=1,
                      sequence_feature=['sum', 'mean', 'max'], include_length=False, task_types=('binary', 'binary'),
                      hash_flag=False, prefix=''):
    feature_columns = []
    model_input = {}

    if 'weight' in sequence_feature:
        feature_columns.append(
            VarLenSparseFeat(SparseFeat(prefix + "weighted_seq", vocabulary_size=2, embedding_dim=embedding_size),
                             maxlen=3, length_name=prefix + "weighted_seq" + "_seq_length",
                             weight_name=prefix + "weight"))
        s_input, s_len_input = gen_sequence(
            2, 3, sample_size)

        model_input[prefix + "weighted_seq"] = s_input
        model_input[prefix + 'weight'] = np.random.randn(sample_size, 3, 1)
        model_input[prefix + "weighted_seq" + "_seq_length"] = s_len_input
        sequence_feature.pop(sequence_feature.index('weight'))

    for i in range(sparse_feature_num):
        dim = np.random.randint(1, 10)
        feature_columns.append(SparseFeat(prefix + 'sparse_feature_' + str(i), dim, embedding_size, dtype=torch.int32))
    for i in range(dense_feature_num):
        feature_columns.append(DenseFeat(prefix + 'dense_feature_' + str(i), 1, dtype=torch.float32))
    for i, mode in enumerate(sequence_feature):
        dim = np.random.randint(1, 10)
        maxlen = np.random.randint(1, 10)
        feature_columns.append(
            VarLenSparseFeat(SparseFeat(prefix + 'sequence_' + mode, vocabulary_size=dim, embedding_dim=embedding_size),
                             maxlen=maxlen, combiner=mode))

    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            model_input[fc.name] = np.random.randint(0, fc.vocabulary_size, sample_size)
        elif isinstance(fc, DenseFeat):
            model_input[fc.name] = np.random.random(sample_size)
        else:
            s_input, s_len_input = gen_sequence(
                fc.vocabulary_size, fc.maxlen, sample_size)
            model_input[fc.name] = s_input
            if include_length:
                fc.length_name = prefix + "sequence_" + str(i) + '_seq_length'
                model_input[prefix + "sequence_" + str(i) + '_seq_length'] = s_len_input

    y_list = []  # multi label
    for task in task_types:
        if task == 'binary':
            y = np.random.randint(0, 2, sample_size)
            y_list.append(y)
        else:
            y = np.random.random(sample_size)
            y_list.append(y)
    y_list = np.array(y_list).transpose()  # (sample_size, num_tasks)

    return model_input, y_list, feature_columns


def check_mtl_model(model, model_name, x, y_list, task_types, check_model_io=True):
    '''
    compile model,train and evaluate it,then save/load weight and model file.
    :param model:
    :param model_name:
    :param x:
    :param y_list: mutil label of y
    :param task_types:
    :param check_model_io:
    :return:
    '''
    loss_list = []
    for task_type in task_types:
        if task_type == 'binary':
            loss_list.append('binary_crossentropy')
        elif task_type == 'regression':
            loss_list.append('mae')
    print('loss:', loss_list)

    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, verbose=1, patience=0, mode='max')
    model_checkpoint = ModelCheckpoint(filepath='model.ckpt', monitor='val_acc', verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False, mode='max', period=1)

    model.compile('adam', loss_list, metrics=['binary_crossentropy', 'acc'])
    model.fit(x, y_list, batch_size=100, epochs=1, validation_split=0.5, callbacks=[early_stopping, model_checkpoint])

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


def get_device(use_cuda=True):
    device = 'cpu'
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    return device
