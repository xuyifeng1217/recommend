import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from collections import namedtuple

DenseFeat = namedtuple("DenseFeat", ['name', 'dimension'])
SparseFeat = namedtuple("SparseFeat", ['name', 'vocabulary_size', 'embedding_dim'])


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


def build_input_layers(feature_columns):
    dense_input_dict, sparse_input_dict = {}, {}

    for feat in feature_columns:
        if isinstance(feat, DenseFeat):
            dense_input_dict[feat.name] = keras.layers.Input(shape=(feat.dimension,), name=feat.name)
        elif isinstance(feat, SparseFeat):
            sparse_input_dict[feat.name] = keras.layers.Input(shape=(1,), name=feat.name)
    return dense_input_dict, sparse_input_dict


def build_embedding_layers(feature_columns, input_layers_dict, is_linear):
    embedding_layers_dict = {}

    # 将特征中的sparse特征筛选出来
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []

    if is_linear:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = keras.layers.Embedding(fc.vocabulary_size, 1, name='1d_emb_' + fc.name)
    else:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = keras.layers.Embedding(fc.vocabulary_size, fc.embedding_dim,
                                                                    name='kd_emb_' + fc.name)

    return embedding_layers_dict


def get_linear_logits(dense_input_dict, sparse_input_dict, sparse_feature_columns):
    # 将所有的dense特征进行拼接，再fc
    concat_dense_inputs = keras.layers.Concatenate(axis=1)(list(dense_input_dict.values()))
    dense_logits_output = keras.layers.Dense(1)(concat_dense_inputs)

    # 将linear部分的sparse特征进行一维的embedding
    linear_embedding_layers = build_embedding_layers(sparse_feature_columns, sparse_input_dict, is_linear=True)

    # 将一维的embedding拼接，embedding操作会使得维度+1，所以需要Flatten操作
    sparse_1d_embed = []
    for fc in sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]  # batch * 1
        embed = keras.layers.Flatten()(linear_embedding_layers[fc.name](feat_input))  # batch * 1*1 -> batch *1
        sparse_1d_embed.append(embed)

    # 因为embedding中查询得到的weight就是sparse特征进行onehot之后1位置所对应的weight，所以之后就不需要再接一个fc层了，
    sparse_logits_output = keras.layers.Add()(sparse_1d_embed)

    linear_part = keras.layers.Add()([dense_logits_output, sparse_logits_output])
    return linear_part


class BiInteractionPooling(keras.layers.Layer):
    def __init__(self):
        super(BiInteractionPooling, self).__init__()

    def call(self, inputs):
        # 优化后的公式与FM很像为：0.5*(和的平方-平方的和)，但是得到k维向量后不进行求和，就保持向量到dnn部分去
        concated_embeds_value = inputs  # batch * n * k

        square_of_sum = tf.square(tf.reduce_sum(concated_embeds_value, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keepdims=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, input_shape[2])


def get_bi_interaction_pooling_output(sparse_input_dict, sparse_feature_columns, dnn_embedding_layers):
    # 只考虑sparse的二阶交叉，将所有的embedding拼接到一起
    sparse_kd_embed = []
    for fc in sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]
        embed = dnn_embedding_layers[fc.name](feat_input)  # batch_size * 1 * k
        sparse_kd_embed.append(embed)

    # 将所有sparse特征的embedding拼接，得到(n, k)的矩阵
    concat_sparse_kd_embed = keras.layers.Concatenate(axis=1)(sparse_kd_embed)  # batch * n * k
    pooling_out = BiInteractionPooling()(concat_sparse_kd_embed)

    return pooling_out


def get_dnn_logits(pooling_out):
    dnn_out = keras.layers.Dropout(0.5)(keras.layers.Dense(1024, activation='relu')(pooling_out))
    dnn_out = keras.layers.Dropout(0.3)(keras.layers.Dense(512, activation='relu')(dnn_out))
    dnn_out = keras.layers.Dropout(0.1)(keras.layers.Dense(256, activation='relu')(dnn_out))

    dnn_logits = keras.layers.Dense(1)(dnn_out)
    return dnn_logits


def NFM(linear_feature_columns, dnn_feature_columns):
    # 构建输入层，
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns + dnn_feature_columns)

    # linear部分中的sparse特征进行筛选
    linear_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), linear_feature_columns))

    # 构建模型的输入层, 要注意的是模型的输入层不能是字典的形式，只能是列表的形式，所以需要将dict转化为list
    # 而之后我们通过模型调用call函数的时候，将实例输入的时候可以传入字典数据，此时dict的数据会根据字典的key来和
    # Input()层的name进行一一对应传入
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())

    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_sparse_feature_columns)

    # 构建维度为k的embedding层集合，该embedding用于FM交叉部分和DNN的输入部分
    embedding_layers = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False)

    # 将输入到dnn中的sparse特征筛选出来
    dnn_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))

    pooling_output = get_bi_interaction_pooling_output(sparse_input_dict, dnn_sparse_feature_columns, embedding_layers)

    pooling_output = keras.layers.BatchNormalization()(pooling_output)

    dnn_logits = get_dnn_logits(pooling_output)

    output_logits = keras.layers.Add()([linear_logits, dnn_logits])

    output_layers = keras.layers.Activation("sigmoid")(output_logits)

    model = keras.models.Model(inputs=input_layers, outputs=output_layers)

    return model


if __name__ == '__main__':

    path = '/Users/yifengxu/PycharmProjects/team-learning-rs-master/DeepRecommendationModel/代码/data/criteo_sample.txt'
    data = pd.read_csv(path)

    columns = data.columns.values
    dense_features = [feat for feat in columns if 'I' in feat]
    sparse_features = [feat for feat in columns if "C" in feat]

    train_data = data_process(data, dense_features, sparse_features)
    train_data['label'] = data['label']

    # 将特征分组为linear和dnn部分，并进行分组标记
    linear_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4) for
                              feat in sparse_features] + [DenseFeat(feat, dimension=1) for feat in dense_features]

    dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4) for
                           feat in sparse_features] + [DenseFeat(feat, dimension=1) for feat in dense_features]

    # 构建NFM模型

    model = NFM(linear_feature_columns, dnn_feature_columns)

    # model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['binary_crossentropy', keras.metrics.AUC(name='auc')])

    train_model_input = {name: data[name] for name in dense_features + sparse_features}
    model.fit(train_model_input, train_data['label'].values,
              batch_size=64, epochs=5, validation_split=0.1)