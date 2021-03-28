#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 下午2:04
# @Author  : xuyifeng
# @File    : test.py
# @Software: PyCharm

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np

# model = keras.Sequential(name='my_sequential')
# model.add(keras.layers.Input(shape=(250,250,3)))
# model.add(layers.Conv2D(32, 5, strides=2, activation='relu')) #None*123*123*32
# model.add(layers.Conv2D(32, 3, activation='relu')) # None, 121, 121,32
# model.add(layers.MaxPool2D(3)) #None, 40, 40, 32
#
#
# model.add(layers.Conv2D(32, 3, activation='relu')) # None, 38,38,32
# model.add(layers.Conv2D(32, 3, activation='relu')) # None, 36, 36, 32
# model.add(layers.MaxPool2D(3)) # None, 12, 12, 32

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()



inputs = layers.Input(shape=(784,))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10)(x)

model = keras.models.Model(inputs=inputs, outputs=outputs, name='mnist_model')

# keras.utils.plot_model(model, 'my_first_model.png')

x_train = x_train.reshape(-1, 784).astype('float32')/255
x_test = x_test.reshape(-1, 784).astype('float32')/255

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(learning_rate=0.002),
              metrics=keras.metrics.SparseCategoricalAccuracy())

history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)

test_score = model.evaluate(x_test, y_test, verbose=2)
print('test loss: {}'.format(test_score[0]))
print('test accuracy: {}'.format(test_score[1]))

model.save('my_first_model')
del model

model = keras.models.load_model('my_first_model')