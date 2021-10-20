'''This file holds the training loops for the KiU-Net model found in model.py

Class: CISC867-Deep Learning
Author: Henry Lee
Date: 2021-10-07
'''
import tensorflow as tf
from tensorflow import keras
from model import kiunet
import data.add_data as data

x_train = data.import_data_numpy("./data/training/images/processed_labels")
y_train = data.import_data_numpy("./data/training/av/processed_labels")

x_test = data.import_data_numpy("./data/testing/images/processed_labels")
y_test = data.import_data_numpy("./data/testing/av/processed_labels")

# insert import from data_processing #
kiunet_model = kiunet((128,128,3))


kiunet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),    ## learning_rate is different from 3D
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC()])

print(x_train[0].shape)
kiunet_model.fit(x_train, y_train, batch_size=20, epochs=10, validation_data=(x_test, y_test),
verbose=1, callbacks=keras.callbacks.TensorBoard(log_dir="./logs"),    ## log metrics in TensorBoard
validation_split=0.15,    ## uses 15% of training data as validation data
)


