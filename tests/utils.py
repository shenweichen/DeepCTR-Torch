# -*- coding: utf-8 -*-
import os

import numpy as np
import torch
from numpy.testing import assert_allclose

from deepctr_torch.inputs import DenseFeat, SparseFeat, VarLenSparseFeat

SAMPLE_SIZE = 64


def gen_sequence(dim, max_len, sample_size):
    return np.array([np.random.randint(0, dim, max_len) for _ in range(sample_size)]), np.random.randint(1, max_len + 1, sample_size)



def get_test_data(sample_size=1000, sparse_feature_num=1, dense_feature_num=1, sequence_feature=('sum', 'mean', 'max'),
                  classification=True, include_length=False, hash_flag=False, prefix=''):

    feature_columns = []

    for i in range(sparse_feature_num):
        dim = np.random.randint(1, 10)
        feature_columns.append(SparseFeat(
            prefix+'sparse_feature_'+str(i), dim, hash_flag, torch.int32))
    for i in range(dense_feature_num):
        feature_columns.append(
            DenseFeat(prefix+'dense_feature_'+str(i), 1, torch.float32))
    for i, mode in enumerate(sequence_feature):
        dim = np.random.randint(1, 10)
        maxlen = np.random.randint(1, 10)
        feature_columns.append(
            VarLenSparseFeat(prefix+'sequence_' + str(i), dim, maxlen, mode))

    model_input = []
    sequence_input = []
    sequence_len_input = []
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            model_input.append(np.random.randint(0, fc.dimension, sample_size))
        elif isinstance(fc, DenseFeat):
            model_input.append(np.random.random(sample_size))
        else:
            s_input, s_len_input = gen_sequence(
                fc.dimension, fc.maxlen, sample_size)
            sequence_input.append(s_input)
            sequence_len_input.append(s_len_input)

    if classification:
        y = np.random.randint(0, 2, sample_size)
        while sum(y) < 0.3*sample_size:
            y = np.random.randint(0, 2, sample_size)
    else:
        y = np.random.random(sample_size)

    x = model_input + sequence_input
    if include_length:
        for i, mode in enumerate(sequence_feature):
            dim = np.random.randint(1, 10)
            maxlen = np.random.randint(1, 10)
            feature_columns.append(
                SparseFeat(prefix+'sequence_' + str(i)+'_seq_length', 1, embedding=False))

        x += sequence_len_input

    return x, y, feature_columns


def layer_test(layer_cls, kwargs = {}, input_shape=None, 
               input_dtype=torch.float32, input_data=None, expected_output=None, 
               expected_output_shape=None, expected_output_dtype=None, fixed_batch_size=False):
    '''check layer is valid or not
    
    :param layer_cls:
    :param input_shape:
    :param input_dtype:
    :param input_data:
    :param expected_output:
    :param expected_output_dtype:
    :param fixed_batch_size:

    :return: output of the layer
    '''    
    if input_data is None:
        # generate input data
        if not input_shape:
            raise ValueError("input shape should not be none")
        
        input_data_shape = list(input_shape)
        for i, e in enumerate(input_data_shape):
            if e is None:
                input_data_shape[i] = np.random.randint(1, 4)
        
        if all(isinstance(e, tuple) for e in input_data_shape):
            input_data = []
            for e in input_data_shape:
                rand_input = (10 * np.random.random(e))
                input_data.append(rand_input)
        else:
            rand_input = 10 * np.random.random(input_data_shape)
            input_data = rand_input

    else:
        # use input_data to update other parameters
        if input_shape is None:
            input_shape = input_data.shape
    
    if expected_output_dtype is None:
        expected_output_dtype = input_dtype
    
    # layer initialization
    layer = layer_cls(**kwargs)
    
    if fixed_batch_size:
        input = torch.tensor(input_data.unsqueeze(0), dtype=input_dtype)
    else:
        input = torch.tensor(input_data, dtype=input_dtype)
    
    # calculate layer's output
    output = layer(input)

    if not output.dtype == expected_output_dtype:
        raise AssertionError("layer output dtype does not match with the expected one")
    
    if not expected_output_shape:
            raise ValueError("expected output shape should not be none")

    actual_output_shape = output.shape
    for expected_dim, actual_dim in zip(expected_output_shape, actual_output_shape):
        if expected_dim is not None:
            if not expected_dim == actual_dim:
                raise AssertionError(f"expected_dim:{expected_dim}, actual_dim:{actual_dim}")
    
    if expected_output is not None:
        # check whether output equals to expected output
        assert_allclose(output, expected_output, rtol=1e-3)
    
    return output



def check_model(model, model_name, x, y, check_model_io=True):
    '''compile model,train and evaluate it,then save/load weight and model file.
    
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
