import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

path = '/Users/yifengxu/PycharmProjects/team-learning-rs-master/DeepRecommendationModel/代码/data/criteo_sample.txt'
data = pd.read_csv(path)

data.head()

from sklearn.preprocessing import LabelEncoder


def data_process(data_df, dense_features, sparse_featues):
    data_df[dense_features] = data_df[dense_features].fillna(0.0)
    for f in dense_features:
        data_df[f] = data_df[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)

    data_df[sparse_featues] = data_df[sparse_featues].fillna("-1")
    for f in sparse_featues:
        lbe = LabelEncoder()
        data_df[f] = lbe.fit_transform(data_df[f])
    return data_df[dense_features + sparse_featues]


columns = data.columns
dense_features = [feat for feat in columns if 'I' in feat]
sparse_features = [feat for feat in columns if 'C' in feat]
train_data = data_process(data, dense_features, sparse_features)
train_data['label'] = data['label']

train_data.head()

from collections import namedtuple

SparseFeat = namedtuple("SparseFeat", ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple("DenseFeat", ['name', 'dimension'])
VarLenSparseFeat = namedtuple("VarLenSparseFeat", ['name', 'vocabulary_size', 'embedding_dim', 'maxlen'])

dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4) for feat in
                       sparse_features] + \
                      [DenseFeat(feat, dimension=1) for feat in dense_features]


def build_input_layers(feature_columns):
    dense_input_dict, sparse_input_dict = {}, {}
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            sparse_input_dict[fc.name] = keras.layers.Input(shape=(1,), name=fc.name)
        elif isinstance(fc, DenseFeat):
            dense_input_dict[fc.name] = keras.layers.Input(shape=(fc.dimension,), name=fc.name)
    return dense_input_dict, sparse_input_dict


def build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear):
    embedding_layers_dict = {}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    if is_linear:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = keras.layers.Embedding(fc.vocabulary_size + 1, 1, name='1d_emb_' + fc.name)
    else:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = keras.layers.Embedding(fc.vocabulary_size + 1, fc.embedding_dim,
                                                                    name='kd_emb_' + fc.name)
    return embedding_layers_dict


def concat_embedding_list(feature_columns, input_layer_dict, embedding_layer_dict, flatten=False):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    embedding_list = []
    for fc in sparse_feature_columns:
        _input = input_layer_dict[fc.name]  # 获得输入层 B * 1
        _embed = embedding_layer_dict[fc.name]  # vocabulary_size * dim
        embed = _embed(_input)  # B *1* dim
        if flatten:
            embed = keras.layers.Flatten()(embed)
        embedding_list.append(embed)
    return embedding_list


class ResidualBlock(keras.layers.Layer):
    def __init__(self, units):
        super(ResidualBlock, self).__init__()
        self.units = units

    def build(self, input_shape):
        out_dim = input_shape[-1]
        self.dnn1 = keras.layers.Dense(self.units, activation='relu')
        self.dnn2 = keras.layers.Dense(out_dim, activation='relu')

    def call(self, inputs):
        x = inputs
        x = self.dnn1(x)
        x = self.dnn2(x)
        x = keras.layers.Activation('relu')(inputs + x)
        return x


def get_dnn_logits(dnn_inputs, block_nums=3):
    dnn_out = dnn_inputs
    for i in range(block_nums):
        dnn_out = ResidualBlock(64)(dnn_out)
    dnn_logits = keras.layers.Dense(1, activation='sigmoid')(dnn_out)
    return dnn_logits


def DeepCrossing(dnn_feature_columns):
    dense_input_dict, sparse_input_dict = build_input_layers(dnn_feature_columns)
    # 构建输入层，模型的输入不能是字典形式，应该讲字典形式转化为列表的形式
    # 这里实际的输入与keras.layers.Input()层对应，是通过模型输入时字典的key与对应Input层的name相对应
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())

    embedding_layer_dict = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False)
    dense_dnn_list = list(dense_input_dict.values())
    dense_dnn_inputs = keras.layers.Concatenate(axis=1)(dense_dnn_list)  # B * n

    sparse_dnn_list = concat_embedding_list(dnn_feature_columns, sparse_input_dict, embedding_layer_dict, flatten=True)
    sparse_dnn_inputs = keras.layers.Concatenate(axis=1)(sparse_dnn_list)  # B * (dim*m)

    # 将dense特征和sparse特征拼接在一起
    dnn_inputs = keras.layers.Concatenate(axis=1)([dense_dnn_inputs, sparse_dnn_inputs])  # B * (n+dim*m)
    # 输入到dnn中，需要提前定义几个残差块
    output_layer = get_dnn_logits(dnn_inputs, block_nums=3)
    model = keras.Model(input_layers, output_layer)
    return model


history = DeepCrossing(dnn_feature_columns)

# history.summary()

history.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['binary_crossentropy', tf.keras.metrics.AUC(name='auc')])

train_model_input = {name: train_data[name] for name in dense_features + sparse_features}
history.fit(train_model_input, train_data['label'].values,
            batch_size=64, epochs=5, validation_split=0.2)