#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/27 上午12:49
# @Author  : xuyifeng
# @File    : DIN.py
# @Software: PyCharm


import tensorflow as tf
from tensorflow import keras
from collections import namedtuple
import pandas as pd
import numpy as np

DenseFeat = namedtuple("DenseFeat", ['name', 'dimension'])
SparseFeat = namedtuple("SparseFeat", ['name', 'vocabulary_size', 'embedding_dim'])
VarLenSparseFeat = namedtuple("VarLenSparseFeat", ['name', 'vocabulary_size', 'embedding_dim', 'maxlen'])


def build_input_layers(feature_columns):
    # 构建输入层
    # 将输入的数据转换成字典的形式，定义输入层的时候让输入层的name和字典中特征的key一致，就可以使得输入的数据和对应的Input层对应
    input_layer_dict = {}
    for fc in feature_columns:
        if isinstance(fc, DenseFeat):
            input_layer_dict[fc.name] = keras.layers.Input(shape=(fc.dimension,), name=fc.name)
        elif isinstance(fc, SparseFeat):
            input_layer_dict[fc.name] = keras.layers.Input(shape=(1,), name=fc.name)
        elif isinstance(fc, VarLenSparseFeat):
            input_layer_dict[fc.name] = keras.layers.Input(shape=(fc.maxlen,), name=fc.name)
    return input_layer_dict


def concat_input_list(input_list):
    feature_nums = len(input_list)
    if feature_nums > 1:
        return keras.layers.Concatenate(axis=1)(input_list)
    elif feature_nums == 1:
        return input_list[0]
    else:
        return None


def build_embedding_layers(feature_columns, input_layer_dict):
    # 构建embedding层
    embedding_layer_dict = {}

    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            embedding_layer_dict[fc.name] = keras.layers.Embedding(fc.vocabulary_size, fc.embedding_dim,
                                                                   name='emb_' + fc.name)
        elif isinstance(fc, VarLenSparseFeat):
            embedding_layer_dict[fc.name] = keras.layers.Embedding(fc.vocabulary_size + 1, fc.embedding_dim,
                                                                   name='emb_' + fc.name, mask_zero=True)

    return embedding_layer_dict


def concat_embedding_list(feature_columns, input_layer_dict, embedding_layer_dict, flatten=False):
    # 将所有的Sparse特征embedding拼接
    embedding_list = []
    for fc in feature_columns:
        print(fc.name)
        _input = input_layer_dict[fc.name]  # 输入层
        embed = embedding_layer_dict[fc.name](_input)  # batch*1 -> batch * 1 * embed_dim

        if flatten:
            embed = keras.layers.Flatten()(embed)
        embedding_list.append(embed)
    return embedding_list


def embedding_lookup(feature_columns, input_layer_dict, embedding_layer_dict):
    embedding_list = []
    for fc in feature_columns:
        _input = input_layer_dict[fc]
        _embed = embedding_layer_dict[fc]
        embed = _embed(_input)
        embedding_list.append(embed)
    return embedding_list


class Dice(keras.layers.Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = keras.layers.BatchNormalization(center=False, scale=False)

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(input_shape[-1],), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)
        return self.aplha * (1.0 - x_p) * x + x_p * x


class LocalActivationUnit(keras.layers.Layer):
    def __init__(self, hidden_units=(256, 128, 64), activation='prelu'):
        super(LocalActivationUnit, self).__init__()
        self.hidden_units = hidden_units
        self.linear = Dense(1)
        self.dnn = [keras.layers.Dense(unit, activation=keras.layers.PReLU() if activation == 'prelu' else Dice())
                    for unit in hidden_units]

    def call(self, inputs):
        # query:b*1*emb_dim, keys:b*len*emb_dim
        query, keys = inputs

        # 获取序列长度
        keys_len = keys.get_shape()[1]

        # 沿着seq_len方向进行复制，可以方便每个query与对应的key做操作
        queries = tf.tile(query, multiples=[1, keys_len, 1])  # b*len*emb_dim

        # 将特征进行拼接
        att_input = keras.layers.Concatenate(axis=-1)(
            [queries, keys, queries - keys, queries * keys])  # b*len*(4*emb_dim)

        #
        att_out = att_input
        for fc in self.dnn:
            att_out = fc(att_out)  # b*len*att_out
        att_out = self.linear(att_out)  # b*len*1
        att_out = keras.layers.Flatten()(att_out)  # b*len

        return att_out


class AttentionPoolingLayer(keras.layers.Layer):
    def __init__(self, att_hidden_units=(256, 128, 64)):
        super(AttentionPoolingLayer, self).__init__()
        self.att_hidden_units = att_hidden_units
        self.local_att = LocalActivationUnit(self.att_hidden_units)

    def call(self, inputs):
        # queries: batch * 1 * emb_dim, keys: batch * len * emb_dim
        queries, keys = inputs

        # 获取行为序列embedding的mask矩阵，将embedding矩阵中的非零元素设置成True
        key_masks = tf.not_equal(keys[:, :, 0], 0)  # batch * len

        # 获取行为序列中每个商品对应的注意力权重
        attention_score = self.local_att([queries, keys])  # 通过FeedForward学习的，我吐血，这么粗暴

        paddings = tf.zeros_like(attention_score)  # b*len

        #
        outputs = tf.where(key_masks, attention_score, paddings)  # b*len

        outputs = tf.expand_dims(outputs, axis=1)  # b*1*len

        # keys: b*len*emb_dim
        outputs = tf.matmul(outputs, keys)  # b*1*emb_dim
        outputs = tf.squeeze(outputs, axis=1)  # b*emb_dim

        return outputs


class Dice(keras.layers.Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = keras.layers.BatchNormalization(center=False, scale=False)

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(input_shape[-1],), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)

        return self.alpha * (1.0 - x_p) * x + x_p * x


def get_dnn_logits(dnn_input, hidden_units=(200, 80), activation='prelu'):
    dnns = [keras.layers.Dense(unit, activation=PReLU() if activation == 'prelu' else Dice()) for unit in hidden_units]

    dnn_out = dnn_input
    for dnn in dnns:
        dnn_out = dnn(dnn_out)

    # 获取logits
    dnn_logits = keras.layers.Dense(1, activation='sigmoid')(dnn_out)

    return dnn_logits


def DIN(feature_columns, behavior_feature_list, behavior_seq_feature_list):
    # 构建输入层
    input_layer_dict = build_input_layers(feature_columns)
    #     print(input_layer_dict)

    # 将Input层转化成列表的形式作为model的输入
    input_layers = list(input_layer_dict.values())

    # 筛选出特征中的sparse特征和dense特征，方便单独处理
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns))

    # 获取dense feature
    dnn_dense_input = []
    for fc in dense_feature_columns:
        dnn_dense_input.append(input_layer_dict[fc.name])

    # 将所有的dense特征进行拼接
    dnn_dense_input = concat_input_list(dnn_dense_input)

    # 构建embedding字典
    embedding_layer_dict = build_embedding_layers(feature_columns, input_layer_dict)
    print(embedding_layer_dict)

    # 将所有sparse特征进行embedding后，因为后面需要进行拼接后加入到fc层，所以需要Flatten()
    dnn_sparse_embed_input = concat_embedding_list(sparse_feature_columns, input_layer_dict, embedding_layer_dict,
                                                   flatten=True)

    # 将所有的sparse特征的embedding进行拼接
    dnn_sparse_input = concat_input_list(dnn_sparse_embed_input)

    # 获取当前的行为特征（movie）的embedding，这里可能会产生多个行为，即行为序列，所以需要使用列表将其放在一起
    query_embed_list = embedding_lookup(behavior_feature_list, input_layer_dict, embedding_layer_dict)

    # 获取行为序列（movie_id序列，hist_movie_id）对应的embedding，
    keys_embed_list = embedding_lookup(behavior_seq_feature_list, input_layer_dict, embedding_layer_dict)

    # 使用attention机制将历史movie_id序列进行池化
    dnn_seq_input_list = []
    for i in range(len(keys_embed_list)):
        seq_emb = AttentionPoolingLayer()([query_embed_list[i], keys_embed_list[i]])
        dnn_seq_input_list.append(seq_emb)

    dnn_seq_input = concat_input_list(dnn_seq_input_list)

    # 将dense特征、sparse特征、通过注意力加权的序列特征进行拼接
    dnn_input = keras.layers.Concatenate(axis=1)([dnn_dense_input, dnn_sparse_input, dnn_seq_input])

    dnn_logits = get_dnn_logits(dnn_input, activation='prelu')

    model = keras.models.Model(input_layers, dnn_logits)

    return model


path = '/Users/yifengxu/PycharmProjects/team-learning-rs-master/DeepRecommendationModel/代码/data/movie_sample.txt'
samples_data = pd.read_csv(path, sep="\t", header=None)
samples_data.columns = ['user_id', 'gender', 'age', 'hist_movie_id', 'hist_len', 'movie_id', 'movie_type_id', 'label']

# samples_data.head()

X = samples_data[["user_id", "gender", "age", "hist_movie_id", "hist_len", "movie_id", "movie_type_id"]]
y = samples_data["label"]

X_train = {'user_id': np.array(X['user_id']),
           'gender': np.array(X['gender']),
           'age': np.array(X['age']),
           'hist_movie_id': np.array([[int(i) for i in line.split(',')] for line in X['hist_movie_id']]),
           'hist_len': np.array(X['hist_len']),
           'movie_id': np.array(X['movie_id']),
           'movie_type_id': np.array(X['movie_type_id'])}
y_train = np.array(y)

feature_columns = [SparseFeat('user_id', max(samples_data['user_id']) + 1, embedding_dim=8),
                   SparseFeat('gender', max(samples_data['gender']) + 1, embedding_dim=8),
                   SparseFeat('age', max(samples_data["age"]) + 1, embedding_dim=8),
                   SparseFeat('movie_id', max(samples_data["movie_id"]) + 1, embedding_dim=8),
                   SparseFeat('movie_type_id', max(samples_data["movie_type_id"]) + 1, embedding_dim=8),
                   DenseFeat('hist_len', 1)
                   ]
feature_columns += [VarLenSparseFeat('hist_movie_id', vocabulary_size=max(samples_data['movie_id']) + 1,
                                     embedding_dim=8, maxlen=50)]

# 特征行为列表，表示的是基本特征
behavior_feature_list = ['movie_id']

# 行为序列特征
behavior_seq_feature_list = ['hist_movie_id']

model = DIN(feature_columns, behavior_feature_list, behavior_seq_feature_list)

model.compile('adam', 'binary_crossentropy')

model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.2, )