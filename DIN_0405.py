#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/5 下午9:38
# @Author  : xuyifeng
# @File    : DIN_0405.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from collections import namedtuple

SparseFeat = namedtuple("SparseFeat", ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'maxlen'])

def build_input_layer(feature_columns):
    input_layer_dict = {}

    for fc in feature_columns:
        if isinstance(fc, DenseFeat):
            input_layer_dict[fc.name] = keras.Input(shape=(fc.dimension,), name=fc.name)
        elif isinstance(fc, SparseFeat):
            input_layer_dict[fc.name] = keras.Input(shape=(1, ), name=fc.name)
        elif isinstance(fc, VarLenSparseFeat):
            input_layer_dict[fc.name] = keras.Input(shape=(fc.maxlen, ), name=fc.name)

    return input_layer_dict


def concat_input_list(input_list):
    n = len(input_list)
    if n > 1:
        return keras.layers.Concatenate(axis=1)(input_list)
    elif n == 1:
        return input_list[0]
    else:
        return None


def build_embedding_layers(feature_columns):
    embedding_layer_dict = {}

    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            embedding_layer_dict[fc.name] = keras.layers.Embedding(fc.vocabulary_size, fc.embedding_dim,
                                                                   name='emb_'+fc.name)
        elif isinstance(fc, VarLenSparseFeat):
            embedding_layer_dict[fc.name] = keras.layers.Embedding(fc.vocabulary_size+1, fc.embedding_dim,
                                                                   name='emb_'+fc.name, mask_zero=True)

    return embedding_layer_dict


def get_embedding_list(sparse_feature_columns, input_layer_dict, embedding_layer_dict, flatten=False):

    embedding_list = []
    for fc in sparse_feature_columns:
        _input = input_layer_dict[fc.name]
        _emb = embedding_layer_dict[fc.name]
        emb = _emb(_input)

        if flatten:
            emb = keras.layers.Flatten()(emb)
        embedding_list.append(emb)
    return embedding_list


def embedding_lookup(feature_columns, input_layer_dict, embedding_layer_dict):
    query_embed_list = []

    for fc in feature_columns:
        _input = input_layer_dict[fc]
        _embed = embedding_layer_dict[fc]
        embed = _embed(_input)
        query_embed_list.append(embed)

    return query_embed_list


# class Dice():




class LocalActivationUnit(keras.layers.Layer):
    def __init__(self, hidden_units=(256, 128, 64), activation='prelu'):
        super(LocalActivationUnit, self).__init__()
        self.hidden_units = hidden_units
        self.dnn = [keras.layers.Dense(unit, activation=keras.layers.PReLU()) for unit in hidden_units]
        self.linear = keras.layers.Dense(1)

    def call(self, inputs):
        # q:None, 1, emb; k:None, seq, emb
        q, k = inputs
        k_len = k.shape[1]

        q = tf.tile(q, multiples=[1,k_len,1])

        # 将特征进行拼接, None, seq, 4*emb
        att_input = keras.layers.Concatenate(axis=-1)(
            [q, k, q-k, q*k]
        )

        # train
        att_out = att_input
        for fc in self.dnn:
            att_out = fc(att_out) #None, seq, 64

        att_out = self.linear(att_out) # None, seq, 1
        att_out = keras.layers.Flatten()(att_out) # None, seq

        return att_out


class AttentionPoolingLayer(keras.layers.Layer):
    def __init__(self, att_hidden_units=(256, 128, 64)):
        super(AttentionPoolingLayer, self).__init__()
        self.att_hidden_units = att_hidden_units
        self.local_att = LocalActivationUnit(self.att_hidden_units)

    def call(self, inputs):

        # q: None, 1, embed; k: None, seq, embed
        q, k = inputs

        attention_score = self.local_att([q, k]) # None, seq

        k_mask = tf.not_equal(k[:, :, 0], 0) # None, seq
        padding = tf.zeros_like(attention_score) # None, seq

        outputs = tf.where(k_mask, attention_score, padding) # 行为序列只对有行的为进行attention; None, seq
        outputs = tf.expand_dims(outputs, axis=1) # None,1,seq
        outputs = tf.matmul(outputs, k) # None, 1, emb
        outputs = tf.squeeze(outputs, axis=1) # None, emb

        return outputs



def get_dnn_logits(dnn_input, hidden_units=(256, 128, 64), activation='prelu'):
    dnns = [keras.layers.Dense(unit, activation=keras.layers.PReLU()) for unit in hidden_units]

    out = dnn_input
    for fc in dnns:
        out = fc(out)

    dnn_logit = keras.layers.Dense(1, activation='sigmoid')(out)

    return dnn_logit


def DIN(feature_columns, behavior_feature_list, behavior_seq_feature_list):
    # 构建输入层
    input_layer_dict = build_input_layer(feature_columns)

    input_layers = list(input_layer_dict.values())

    # dense特征和sparse特征分开处理\
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))

    # dense特征进行拼接处理
    dnn_dense_input = []
    for fc in dense_feature_columns:
        dnn_dense_input.append(input_layer_dict[fc.name])

    dnn_dense_input = concat_input_list(dnn_dense_input)

    embedding_layer_dict = build_embedding_layers(feature_columns)

    # sparse特征进行embedding
    dnn_sparse_embed_input = get_embedding_list(sparse_feature_columns, input_layer_dict, embedding_layer_dict, flatten=True)

    # sparse特征进行拼接
    dnn_sparse_input = concat_input_list(dnn_sparse_embed_input)

    # dense特征和sparse特征处理好之后，先放着，等用户兴趣特征也就是行为序列特征处理好之后，再一起feed

    query_embed_list = embedding_lookup(behavior_feature_list, input_layer_dict, embedding_layer_dict)

    keys_embed_list = embedding_lookup(behavior_seq_feature_list, input_layer_dict, embedding_layer_dict)

    dnn_seq_input_list = []

    for q, k in zip(query_embed_list, keys_embed_list):
        seq_emb = AttentionPoolingLayer()([q, k])  # None, emb
        dnn_seq_input_list.append(seq_emb)

    dnn_seq_input = concat_input_list(dnn_seq_input_list)

    # 将dense sparse behavior特征拼接
    dnn_input = keras.layers.Concatenate(axis=-1)([dnn_dense_input, dnn_sparse_input, dnn_seq_input])

    dnn_logits = get_dnn_logits(dnn_input, activation='prelu')

    model = keras.models.Model(inputs=input_layers, outputs=dnn_logits)

    return model



path = '/Users/yifengxu/PycharmProjects/team-learning-rs-master/DeepRecommendationModel/代码/data/movie_sample.txt'
samples_data = pd.read_csv(path, sep='\t', header=None)
samples_data.columns = ['user_id', 'gender', 'age', 'hist_movie_id', 'hist_len', 'movie_id', 'movie_type_id', 'label']


X = samples_data[["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id"]]
y = samples_data["label"]

X_train = {
    'user_id': np.array(X['user_id']),
    'gender': np.array(X['gender']),
    'age': np.array(X['age']),
    'hist_movie_id': np.array([[int(x) for x in line.split(',')] for line in X['hist_movie_id']]),
    'hist_len': np.array(X['hist_len']),
    'movie_id': np.array(X['movie_id']),
    'movie_type_id': np.array(X['movie_type_id'])
}

y_train = np.array(y)

feature_columns = [SparseFeat('user_id', vocabulary_size=max(X['user_id'])+1, embedding_dim=8),
                   SparseFeat('gender', vocabulary_size=max(X['gender'])+1, embedding_dim=8),
                   SparseFeat('age', vocabulary_size=max(X['age'])+1, embedding_dim=8),
                   VarLenSparseFeat('hist_movie_id', vocabulary_size=max(X['movie_id'])+1, embedding_dim=8, maxlen=50),
                   DenseFeat('hist_len', dimension=1),
                   SparseFeat('movie_id', vocabulary_size=max(X['movie_id'])+1, embedding_dim=8),
                   SparseFeat('movie_type_id', vocabulary_size=max(X['movie_type_id'])+1, embedding_dim=8)
                   ]

behavior_feature_list = ['movie_id']

behavior_seq_feature_list = ['hist_movie_id']

model = DIN(feature_columns, behavior_feature_list, behavior_seq_feature_list)

model.compile('adam', 'binary_crossentropy')

model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.2)
