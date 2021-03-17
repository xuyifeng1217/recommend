import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import namedtuple


SparseFeat = namedtuple("SparseFeat", ['name', 'vocabulary_size', 'embedding_dim'])
DenseFeat = namedtuple("DenseFeat", ['name', 'dimension'])
VarLenSparseFeat = namedtuple('VarLenSparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'maxlen'])
from sklearn.preprocessing import LabelEncoder


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


path = '/Users/yifengxu/PycharmProjects/team-learning-rs-master/DeepRecommendationModel/代码/data/criteo_sample.txt'
data = pd.read_csv(path)
# 划分dense和sparse特征
columns = data.columns.values
dense_features = [feat for feat in columns if 'I' in feat]
sparse_features = [feat for feat in columns if 'C' in feat]

# 简单的数据预处理
train_data = data_process(data, dense_features, sparse_features)
train_data['label'] = data['label']

# 将特征进行分组，分成linear部分和dnn部分，并将分组后的特征进行标记，使用DenseFeat和SparseFeat
linear_feature_columns = [DenseFeat(feat, dimension=1) for feat in dense_features] + \
                         [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4) for feat in
                          sparse_features]

dnn_feature_columns = [DenseFeat(feat, dimension=1) for feat in dense_features] + \
                      [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4) for feat in
                       sparse_features]


def build_input_layers(feature_columns):
    dense_input_dict, sparse_input_dict = {}, {}
    for feat in feature_columns:
        if isinstance(feat, SparseFeat):
            sparse_input_dict[feat.name] = keras.layers.Input(shape=(1,), name=feat.name)
        elif isinstance(feat, DenseFeat):
            dense_input_dict[feat.name] = keras.layers.Input(shape=(feat.dimension,), name=feat.name)
    return dense_input_dict, sparse_input_dict


def build_embedding_layers(feature_columns, input_layers_dict, is_linear):
    embedding_layers_dict = {}

    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))

    # # 如果是用于线性部分的embedding层，其维度为1，否则维度就是自己定义的embedding维度
    if is_linear:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = keras.layers.Embedding(fc.vocabulary_size, 1, name='1d_emb_' + fc.name)
    else:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = keras.layers.Embedding(fc.vocabulary_size, fc.embedding_dim,
                                                                    name='kd_emb_' + fc.name)
    return embedding_layers_dict


def get_linear_logits(dense_input_dict, sparse_input_dict, sparse_feature_columns):
    # 将所有的dense特征的input层拼接，经过一个fc得到dense特征的logits
    concat_dense_inputs = keras.layers.Concatenate(axis=1)(list(dense_input_dict.values()))
    dense_logits_output = keras.layers.Dense(1)(concat_dense_inputs)

    # 获取linear部分sparse特征的embedding层，这里使用embedding的原因是：
    # 对于linear部分直接将特征进行onehot然后通过一个全连接层，当维度特别大的时候，计算比较慢
    # 使用embedding层的好处就是可以通过查表的方式获取到哪些非零的元素对应的权重，然后在将这些权重相加，效率比较高
    linear_embedding_layers = build_embedding_layers(sparse_feature_columns, sparse_input_dict, is_linear=True)

    sparse_1d_embed = []
    for fc in sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]  # name -> Input()
        embed = keras.layers.Flatten()(linear_embedding_layers[fc.name](feat_input))  # b * 1 * 1 -> b * 1
        sparse_1d_embed.append(embed)

    # embedding中查询得到的权重就是对应onehot向量中一个位置的权重，所以后面不用再接一个全连接了，本身一维的embedding就相当于全连接
    # 只不过是这里的输入特征只有0和1，所以直接向非零元素对应的权重相加就等同于进行了全连接操作(非零元素部分乘的是1)

    sparse_logits_output = keras.layers.Add()(sparse_1d_embed)  # b * 1
    linear_logits = keras.layers.Add()([dense_logits_output, sparse_logits_output])  # b*1
    return linear_logits


def concat_embedding_list(feature_columns, input_layer_dict, embedding_layer_dict, flatten=False):
    # # 将sparse特征筛选出来
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    embedding_list = []
    for fc in sparse_feature_columns:
        _input = input_layer_dict[fc.name]  # 获取输入层
        _embed = embedding_layer_dict[fc.name]  # 获取embedding层，batch*1*dim
        embed = _embed(_input)  # batch * 1 * dim

        if flatten:
            embed = keras.layers.Flatten()(embed)
        embedding_list.append(embed)
    return embedding_list


def get_dnn_logits(dense_input_dict, sparse_input_dict, sparse_feature_columns, dnn_embedding_layers):
    concat_dense_inputs = keras.layers.Concatenate(axis=1)(list(dense_input_dict.values()))  # B x n1 (n表示的是dense特征的维度)
    sparse_kd_embed = concat_embedding_list(sparse_feature_columns, sparse_input_dict, dnn_embedding_layers,
                                            flatten=True)

    concat_sparse_kd_embed = keras.layers.Concatenate(axis=1)(sparse_kd_embed)  # B x n2k  (n2表示的是Sparse特征的维度)

    dnn_input = keras.layers.Concatenate(axis=1)([concat_dense_inputs, concat_sparse_kd_embed])  # B x (n2k + n1)
    dnn_out = keras.layers.Dropout(0.5)(keras.layers.Dense(1024, activation='relu')(dnn_input))
    dnn_out = keras.layers.Dropout(0.3)(keras.layers.Dense(512, activation='relu')(dnn_out))
    dnn_out = keras.layers.Dropout(0.1)(keras.layers.Dense(256, activation='relu')(dnn_out))

    dnn_logits = keras.layers.Dense(1)(dnn_out)
    return dnn_logits


# Wide&Deep 模型的wide部分及Deep部分的特征选择，应该根据实际的业务场景去确定哪些特征应该放在Wide部分，哪些特征应该放在Deep部分
def WideNDeep(linear_feature_columns, dnn_feature_columns):
    # 构建输入层，将所有的特征对应为Input()
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns + dnn_feature_columns)
    #     print(sparse_input_dict)

    # 将linear部分中的sparse特征筛选出来，后面用来将其做1维的embedding
    linear_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), linear_feature_columns))

    # 构建模型的输入层，模型的输入层需要是list形式的Input()，将字典的形式转化为list形式
    # 这里的Input()层都有对应的name，那么实际输入的时候，可以通过传入字典，根据字典的key自动与对应name的Input()层相对应
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())

    # wide部分
    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_sparse_feature_columns)

    # 构建维度为k的embedding层，这里使用字典的形式返回，方便后面搭建模型
    embedding_layers = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False)

    dnn_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))

    # 在Wide&Deep模型中，deep部分的输入是将dense特征和embedding特征拼在一起输入到dnn中
    dnn_logits = get_dnn_logits(dense_input_dict, sparse_input_dict, dnn_sparse_feature_columns, embedding_layers)

    out_logits = keras.layers.Add()([linear_logits, dnn_logits])
    out_layers = keras.layers.Activation("sigmoid")(out_logits)

    model = keras.models.Model(input_layers, out_layers)

    return model


model = WideNDeep(linear_feature_columns, dnn_feature_columns)
# model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['binary_crossentropy', keras.metrics.AUC(name='auc')])
train_model_input = {name: data[name] for name in dense_features + sparse_features}
model.fit(train_model_input, train_data['label'].values,
          batch_size=64, epochs=50, validation_split=0.2)