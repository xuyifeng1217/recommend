#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/18 上午12:56
# @Author  : xuyifeng
# @File    : W&D.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from collections import namedtuple

DenseFeat = namedtuple('DenseFeat', ['name', 'dimension'])
SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])

# 简单处理特征，包括填充缺失值，数值处理，类别编码
def data_process(data_df, dense_features, sparse_features):
    data_df[dense_features] = data_df[dense_features].fillna(0.0)
    for f in dense_features:
        data_df[f] = data_df[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)

    data_df[sparse_features] = data_df[sparse_features].fillna("-1")
    for f in sparse_features:
        lbe = LabelEncoder()
        data_df[f] = lbe.fit_transform(data_df[f])

    return data_df[dense_features + sparse_features]


def build_input_layers(features):
    dense_input_dict, sparse_input_dict = {}, {}
    for feat in features:
        if isinstance(feat, DenseFeat):
            dense_input_dict[feat.name] = keras.layers.Input(shape=(feat.dimension,), name=feat.name)
        elif isinstance(feat, SparseFeat):
            sparse_input_dict[feat.name] = keras.layers.Input(shape=(1,), name=feat.name)

    return dense_input_dict, sparse_input_dict


# def build_embedding_layers(feature_columns, is_linear):
#     embedding_layer_dict = {}
#
#     sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
#
#     if is_linear:
#         for feat in sparse_feature_columns:
#             embedding_layer_dict[feat.name] = keras.layers.Embedding(feat.vocabulary_size, 1, name='1d_'+feat.name)
#     else:
#         for feat in sparse_feature_columns:
#             embedding_layer_dict[feat.name] = keras.layers.Embedding(feat.vocabulary_size, feat.embedding_dim, name='kd_'+feat.name)
#
#     return embedding_layer_dict


# def get_linear_logits(dense_input_dict, sparse_input_dict, sparse_feature_columns):
#     # 连续型特征
#     concat_dense_input = keras.layers.Concatenate(axis=1)(list(dense_input_dict.values()))
#     dense_logit_output = keras.layers.Dense(1)(concat_dense_input)
#
#     # 离散特征
#     linear_embedding_layers = build_embedding_layers(sparse_feature_columns, is_linear=True)
#
#     sparse_1d_embed = []
#     for feat in sparse_feature_columns:
#         feat_input = sparse_input_dict[feat.name]
#         feat_embed = keras.layers.Flatten()(linear_embedding_layers[feat.name](feat_input))  # None*1*1 -> None*1
#         sparse_1d_embed.append(feat_embed)
#
#     sparse_logit_output = keras.layers.Add()(sparse_1d_embed)
#     linear_logit = keras.layers.Add()([dense_logit_output, sparse_logit_output])  # None*1
#     return linear_logit


def build_embedding_layers(feature_columns, is_linear=True):
    embedding_layer_dict = {}

    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    if is_linear:
        for feat in sparse_feature_columns:
            embedding_layer_dict[feat.name] = keras.layers.Embedding(input_dim=feat.vocabulary_size, output_dim=1, name='1d_'+feat.name)
    else:
        for feat in sparse_feature_columns:
            embedding_layer_dict[feat.name] = keras.layers.Embedding(feat.vocabulary_size, feat.embedding_dim, name='kd_'+feat.name)

    return embedding_layer_dict


def get_linear_logits(dense_input_dict, sparse_input_dict, linear_sparse_feature_columns):
    concat_dense_input = keras.layers.Concatenate(axis=1)(list(dense_input_dict.values()))
    dense_logit_output = keras.layers.Dense(1)(concat_dense_input)

    linear_embedding_layers = build_embedding_layers(linear_sparse_feature_columns, is_linear=True)

    sparse_1d_embed = []
    for feat in linear_sparse_feature_columns:
        input_ = sparse_input_dict[feat.name]
        embed_ = linear_embedding_layers[feat.name]
        embed = keras.layers.Flatten()(embed_(input_))  #None, 1
        sparse_1d_embed.append(embed)

    sparse_logit_output = keras.layers.Add()(sparse_1d_embed) # None, 1
    linear_logit = keras.layers.Add()([dense_logit_output, sparse_logit_output]) # None, 1
    return linear_logit


def concat_embedding_list(feature_columns, sparse_input_dict, embedding_layers, flatten=False):
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))

    embedding_list = []
    for feat in sparse_feature_columns:
        input_ = sparse_input_dict[feat.name]
        embed_ = embedding_layers[feat.name]
        embed = embed_(input_)  # None, 1, embed

        if flatten:
            embed = keras.layers.Flatten()(embed)
        embedding_list.append(embed)
    return embedding_list


def get_dnn_logits(dense_input_dict, sparse_input_dict, dnn_sparse_feature_columns, embedding_layers):
    concat_dense_inputs = keras.layers.Concatenate(axis=1)(list(dense_input_dict.values()))  # None, num_dense
    sparse_kd_embed = concat_embedding_list(dnn_sparse_feature_columns, sparse_input_dict, embedding_layers, flatten=True)

    concat_sparse_inputs = keras.layers.Concatenate(axis=1)(sparse_kd_embed)  # None, num_sparse*dim

    dnn_input = keras.layers.Concatenate(axis=1)([concat_dense_inputs, concat_sparse_inputs])
    dnn_out = keras.layers.Dropout(0.5)(keras.layers.Dense(1024, activation='relu')(dnn_input))
    dnn_out = keras.layers.Dropout(0.3)(keras.layers.Dense(512, activation='relu')(dnn_out))
    dnn_out = keras.layers.Dropout(0.1)(keras.layers.Dense(256, activation='relu')(dnn_out))

    dnn_logit = keras.layers.Dense(1)(dnn_out)

    return dnn_logit





def WideAndDeep(linear_feature_columns, dnn_feature_columns):
    # 构建输入层，将所有特征编译为input
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns+dnn_feature_columns)

    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())

    linear_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), linear_feature_columns))

    # wide loss
    linear_logit = get_linear_logits(dense_input_dict, sparse_input_dict, linear_sparse_feature_columns)

    # deep part embedding dict
    embedding_layers = build_embedding_layers(dnn_feature_columns, is_linear=False)

    dnn_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))

    # deep part loss
    dnn_logit = get_dnn_logits(dense_input_dict, sparse_input_dict, dnn_sparse_feature_columns, embedding_layers)

    out_logit = keras.layers.Add()([linear_logit, dnn_logit])
    out_layer = keras.activations.sigmoid(out_logit)

    model = keras.models.Model(inputs=input_layers, outputs=out_layer)
    return model


if __name__ == '__main__':
    path = '/Users/yifengxu/PycharmProjects/team-learning-rs-master/DeepRecommendationModel/代码/data/criteo_sample.txt'
    data = pd.read_csv(path)
    # 划分dense和sparse特征
    columns = data.columns.values
    dense_features = [feat for feat in columns if 'I' in feat]
    sparse_features = [feat for feat in columns if 'C' in feat]

    train_data = data_process(data, dense_features, sparse_features)
    train_data['label'] = data['label']

    # wide部分特征
    linear_feature_columns = [DenseFeat(x, dimension=1) for x in dense_features] + \
                             [SparseFeat(x, vocabulary_size=train_data[x].nunique(),
                                         embedding_dim=4) for x in sparse_features]

    # deep部分特征
    dnn_feature_columns = [DenseFeat(x, dimension=1) for x in dense_features] + \
                          [SparseFeat(x, vocabulary_size=train_data[x].nunique(),
                                         embedding_dim=4) for x in sparse_features]
    tf.random.set_seed(10)
    model = WideAndDeep(linear_feature_columns, dnn_feature_columns)

    model.compile(optimizer='SGD',
                  loss='BinaryCrossentropy',
                  metrics=['BinaryCrossentropy', "AUC"])

    train_model_input = {name: train_data[name].values for name in dense_features+sparse_features}
    model.fit(train_model_input, train_data['label'].values,
              batch_size=32, epochs=50, validation_split=0.2)
    # keras.optimizers

