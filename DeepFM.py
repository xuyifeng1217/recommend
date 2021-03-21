import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from collections import namedtuple

SparseFeat = namedtuple("SparseFeat", ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple("DenseFeat", ['name', 'dimension'])
VarLenSparseFeat = namedtuple("VarLenSparseFeat", ['name', 'vocabulary_size', "embedding_dim", 'maxlen'])


def data_precess(data, dense_features, sparse_features):
    data[dense_features] = data[dense_features].fillna(0.0)
    for f in dense_features:
        data[f] = data[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)
    data[sparse_features] = data[sparse_features].fillna("-1")
    for f in sparse_features:
        lbe = LabelEncoder()
        data[f] = lbe.fit_transform(data[f])
    return data[dense_features + sparse_features]


def build_input_layers(feature_columns):
    #     构建Input层字典，以dense和sparse两种类型字典的形式返回
    dense_input_dict, sparse_input_dict = {}, {}
    for feat in feature_columns:
        if isinstance(feat, SparseFeat):
            sparse_input_dict[feat.name] = keras.layers.Input(shape=(1,), name=feat.name)
        elif isinstance(feat, DenseFeat):
            dense_input_dict[feat.name] = keras.layers.Input(shape=(feat.dimension,), name=feat.name)
    return dense_input_dict, sparse_input_dict


def build_embedding_layers(feature_columns, input_layers_dict, is_linear):
    # 定义一个embedding层对应的字典
    embedding_layers_dict = {}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    # 如果是用于线性部分的embediing层，则其维度为1，否则维度就自己定义
    if is_linear:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = keras.layers.Embedding(fc.vocabulary_size, 1, name='1d_emb_' + fc.name)
    else:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = keras.layers.Embedding(fc.vocabulary_size, fc.embedding_dim,
                                                                    name='kd_emb_' + fc.name)
    return embedding_layers_dict


def get_linear_logits(dense_input_dict, sparse_input_dict, sparse_feature_columns):
    # 将所有的dense特征的input()层，经过一个全连接层得到dense特征的logits
    concat_dense_input = keras.layers.Concatenate(axis=1)(list(dense_input_dict.values()))
    dense_logits_output = keras.layers.Dense(1)(concat_dense_input)

    # 获取linear部分sparse特征的embedding层,这里使用embedding的原因是：
    # 对于Linear部分直接将特征进行onehot然后通过一个全连接层，当维度特别大的时候，计算会比较慢，
    # 使用embedding就可以通过lookup的方式获取到哪些非零的元素对应的权重，然后再将这些权重相加，效率比较高
    linear_embedding_layers = build_embedding_layers(sparse_feature_columns, sparse_input_dict, is_linear=True)

    # 将一维的embedding拼接，这里需要使用一个Flatten层，使维度对应
    sparse_1d_embed = []
    for fc in sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]  # batch * 1
        embed = keras.layers.Flatten()(linear_embedding_layers[fc.name](feat_input))  # b*1 -> b*1*1 -> b*1
        sparse_1d_embed.append(embed)
    # embedding中通过查询得到的权重就是对应onehot向量中那个非零位置的权重，所以后面就不需要再接一个全连接层了，
    # 然后因为类别型特征进行onehot之后的数值只有0和1，所以我们只需要把embedding中查询得到的权重相加就好了
    sparse_logits_output = keras.layers.Add()(sparse_1d_embed)

    # 最终把dense特征和sparse特征对应的logits相加，得到最终Linear的logits
    linear_logits = keras.layers.Add()([dense_logits_output, sparse_logits_output])
    return linear_logits


# $
# \sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_i\cdot v_j>x_i x_j=\frac{1}{2}\sum_{f=1}^k ((\sum_{i=1}^n v_{if}x_i)^2-
# (\sum_{i=1}^n v_{if}^2x_i^2))
# $


# 所以FM层的操作就是，沿着隐向量维度求和(Embedding矩阵沿着n维度求和的平方 - Embedding矩阵的平方沿着n轴求和) * 0.5


class FM_layer(keras.layers.Layer):
    def __init__(self):
        super(FM_layer, self).__init__()

    def call(self, input):
        concated_embeds_value = input  # batch * n * k
        square_of_sum = tf.square(tf.reduce_sum(concated_embeds_value, axis=1, keepdims=True))  # batch * 1* k
        sum_of_square = tf.reduce_sum(tf.square(concated_embeds_value), axis=1, keepdims=True)  # batch * 1 * k
        cross_term = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=2, keepdims=False)  # batch * 1
        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)


def get_fm_logits(sparse_input_dict, sparse_feature_columns, dnn_embedding_layers):
    # 将特征中的sparse特征筛选出来
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), sparse_feature_columns))

    # 只考虑sparse的二阶交叉，将所有的embedding拼接到一起进行计算，
    # 因为类别型数据输入的只有0和1，所以不需要考虑将隐向量和x相乘，直接对隐向量进行操作即可
    sparse_kd_embed = []
    for fc in sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]
        _embed = dnn_embedding_layers[fc.name](feat_input)  # batch*1*k
        sparse_kd_embed.append(_embed)

    # 将所有sparse的embedding拼接起来，得到(n*k)的矩阵, n个特征，k为embedding_dim
    concat_sparse_kd_embed = keras.layers.Concatenate(axis=1)(sparse_kd_embed)  # -> batch * n * k
    fm_cross_out = FM_layer()(concat_sparse_kd_embed)
    return fm_cross_out


def get_dnn_logits(sparse_input_dict, sparse_feature_columns, dnn_embedding_layers):
    # 将特征中的sparse特征筛选出来
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), sparse_feature_columns))

    sparse_kd_embed = []
    for fc in sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]
        _embed = dnn_embedding_layers[fc.name](feat_input)  # batch * 1 * k
        _embed = keras.layers.Flatten()(_embed)  # batch * k
        sparse_kd_embed.append(_embed)

    concat_sparse_kd_embed = keras.layers.Concatenate(axis=1)(sparse_kd_embed)  # batch * (n*k)

    # dnn层，设置几个fc和dropout
    mlp_out = keras.layers.Dropout(0.5)(keras.layers.Dense(256, activation='relu')(concat_sparse_kd_embed))
    mlp_out = keras.layers.Dropout(0.3)(keras.layers.Dense(256, activation='relu')(mlp_out))
    mlp_out = keras.layers.Dropout(0.1)(keras.layers.Dense(256, activation='relu')(mlp_out))
    dnn_out = keras.layers.Dense(1)(mlp_out)
    return dnn_out


def DeepFM(linear_feature_columns, dnn_feature_columns):
    # 构建输入层，即所有特征对应的Input()层，并存到字典中方便后续调用
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns + dnn_feature_columns)

    #     将linear部分特征中的sparse特征筛选出来，后面用来做1维的embedding
    linear_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), linear_feature_columns))

    # 构建模型的输入，模型的输入层不能是字典的形式，需要将字典转化为list的形式
    # 实际输入的时候，是通过输入一个字典，然后字典的key会与Input()层的name相对应匹配
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())
    # linear logits由两部分组成，分别是dense特征的logits和sparse特征的logits
    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_sparse_feature_columns)

    # 构建维度为k的embedding层，这里使用字典的形式返回，方便后面搭建模型
    # embedding层用户构建FM交叉部分和DNN的输入部分
    embedding_layers = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False)

    # 将输入到dnn中的所有sparse特征筛选出来
    dnn_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))

    # 只考虑二阶项
    fm_logits = get_fm_logits(sparse_input_dict, dnn_sparse_feature_columns, embedding_layers)

    # 将所有的embedding都拼接起来，一起输入到dnn
    dnn_logits = get_dnn_logits(sparse_input_dict, dnn_sparse_feature_columns, embedding_layers)

    # linear、FM、dnn的logits相加作为loss
    output_logits = keras.layers.Add()([linear_logits, fm_logits, dnn_logits])
    output_layers = keras.layers.Activation('sigmoid')(output_logits)

    model = keras.models.Model(input_layers, output_layers)
    return model


if __name__ == '__main__':
    path = '/Users/yifengxu/PycharmProjects/team-learning-rs-master/DeepRecommendationModel/代码/data/criteo_sample.txt'
    data = pd.read_csv(path)

    columns = data.columns.values
    dense_features = [x for x in columns if 'I' in x]
    sparse_features = [x for x in columns if "C" in x]

    train_data = data_precess(data, dense_features, sparse_features)
    train_data["label"] = data['label']

    # 将特征分组，分为linear和dnn部分，并将分组之后的特征进行标记，使用SparseFeat和DenseFeat
    linear_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                              for feat in sparse_features] + [DenseFeat(feat, dimension=1)
                                                              for feat in dense_features]

    dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                           for feat in sparse_features] + [DenseFeat(feat, dimension=1)
                                                           for feat in dense_features]

    # 将输入转化为字典形式
    train_model_input = {name: data[name] for name in dense_features + sparse_features}
    model = DeepFM(linear_feature_columns, dnn_feature_columns)
    # model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['binary_crossentropy', keras.metrics.AUC(name='auc')])

    model.fit(train_model_input, train_data['label'].values,
              batch_size=64, epochs=5, validation_split=0.1)