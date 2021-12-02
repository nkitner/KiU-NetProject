'''This file holds the training loops for the KiU-Net model found in model.py

Class: CISC867-Deep Learning
Author: Henry Lee
Date: 2021-10-07
'''
import tensorflow as tf
from tensorflow import keras
from model import kiunet, unet, segnet
from data.add_data import import_data_numpy, import_data_numpy_mask
import numpy as np
import custom_metrics as cm
import datetime

kiunet_model = kiunet((128, 128, 3), 2)

x_train = import_data_numpy("./data/resized/train/img")
y_train = import_data_numpy_mask("./data/resized/train/labelcol")

x_test = import_data_numpy("./data/resized/test/img")
y_test = import_data_numpy_mask("./data/resized/test/labelcol")

y_train_binary = np.copy(y_train)
y_test_binary = np.copy(y_test)

for i in range(len(y_train)):
    y_train_binary[i][y_train_binary[i] > 0] = 1
    y_test_binary[i][y_test_binary[i] > 0] = 1

y_onehot_train = keras.utils.to_categorical(y_train_binary, 2)
y_onehot_test = keras.utils.to_categorical(y_test_binary, 2)

# Create all three models

# Seg-Net model
segnet_model = segnet((128, 128, 3), 2)

segnet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[cm.f1_metric])

# U-Net model
unet_model = unet((128, 128, 3), 2)

unet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[cm.f1_metric])

# KiU-Net model
kiunet_model = kiunet((128, 128, 3), 2)

kiunet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[cm.f1_metric])

### Train each model ###

# Instaniate logging for tensorboard
segnet_train_log_dir = "logs/unet/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
segnet_train_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=segnet_train_log_dir, histogram_freq=1)

segnet_model.fit(x_train, y_onehot_train, batch_size=1, epochs=300,
                 validation_data=(x_test, y_onehot_test), verbose=1,
                 callbacks=[segnet_train_tensorboard_callback])

segnet_model.save('segnet_model.h5', save_format='h5')

# Instaniate logging for tensorboard
unet_train_log_dir = "logs/unet/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
unet_train_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=unet_train_log_dir, histogram_freq=1)

unet_model.fit(x_train, y_onehot_train, batch_size=1, epochs=300,
               validation_data=(x_test, y_onehot_test), verbose=1,
               callbacks=[unet_train_tensorboard_callback])

unet_model.save('unet_model.h5', save_format='h5')

# Instantiate logging for tensorboard
kiunet_train_log_dir = "logs/kiunet/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
kiunet_train_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=kiunet_train_log_dir, histogram_freq=1)


kiunet_model.fit(x_train, y_onehot_train, batch_size=1, epochs=300,
                 validation_data=(x_test, y_onehot_test), verbose=1,
                 callbacks=[kiunet_train_tensorboard_callback])

kiunet_model.save('kiunet_model.h5', save_format='h5')
