'''This file holds the training loops for the KiU-Net model found in model.py

Class: CISC867-Deep Learning
Author: Henry Lee
Date: 2021-10-07
'''
import tensorflow as tf
from tensorflow import keras
from model import kiunet
from data.add_data import import_data_numpy

kiunet_model = kiunet((128, 128, 3))

x_train = import_data_numpy("./data/resized/train/img")
y_train = import_data_numpy("./data/resized/train/labelcol")

x_test = import_data_numpy("./data/resized/test/img")
y_test = import_data_numpy("./data/resized/test/labelcol")

kiunet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),    ## learning_rate is different from 3D
    loss="binary_crossentropy",  ##maybe categorical_crossentropy if labels one-hot encoded
    metrics=[keras.metrics.Accuracy()])#check metric, should be 4 classes

kiunet_model.fit(x_train, y_train, batch_size=1, epochs=400,
                 validation_data=(x_test, y_test), verbose=1,    ## validation_data = test_data, no early stopping
                 callbacks=keras.callbacks.TensorBoard(log_dir="./logs"))    ## log metrics in TensorBoard

## Save using SavedModel format
kiunet_model.save('kiUnet_model_pb')

## Save again using H5 format
kiunet_model.save('kiUnet_model2_h5', save_format='h5')