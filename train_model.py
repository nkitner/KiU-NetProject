'''This file holds the training loops for the KiU-Net model found in model.py

Class: CISC867-Deep Learning
Author: Henry Lee
Date: 2021-10-07
'''
import tensorflow as tf
from tensorflow import keras
from model import kiunet
from data.add_data import import_data

x_train = import_data("./data/training/images/")
y_train = import_data("./data/training/av/")

x_test = import_data("./data/testing/images/")
y_test = import_data("./data/testing/av/")

# insert import from data_processing #
kiunet_model = kiunet((128,128,3))


kiunet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),    ## learning_rate is different from 3D
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.AUC()])

kiunet_model.fit(x_train, y_train, batch_size=1, epochs=10,
# verbose='auto',
# callbacks=keras.callbacks.TensorBoard(log_dir="./logs"),    ## log metrics in TensorBoard
# validation_split=0.15,    ## uses 15% of training data as validation data
)


