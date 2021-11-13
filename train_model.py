'''This file holds the training loops for KiU-Net and
U-Net found in model.py

Author: Henry Lee. Nicole Kitner
Date: 2021-10-07
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from model import kiunet_model, unet_model
from import_data import import_data_numpy, import_data_numpy_mask
import numpy as np
import os



# Import data
data_dir = "./data/resized"

x_train = import_data_numpy(os.path.join(data_dir, "/train/img"))
y_train = import_data_numpy_mask(os.path.join(data_dir, "/train/labelcol"))
x_test = import_data_numpy(os.path.join(data_dir, "/test/img"))
y_test = import_data_numpy_mask(os.path.join(data_dir, "/test/labelcol"))

# Convert data to binary masks
y_train_binary = np.copy(y_train)
y_test_binary = np.copy(y_test)

for i in range(len(y_train)):
  y_train_binary[i][y_train_binary[i] > 0] = 1  
  y_test_binary[i][y_test_binary[i] > 0] = 1 

y_onehot_train = keras.utils.to_categorical(y_train_binary, 2)
y_onehot_test = keras.utils.to_categorical(y_test_binary, 2)



# Compile KiU-Net
kiunet_model_a1 = kiunet((128,128,3), 2)
m = tf.keras.metrics.MeanIoU(num_classes=256)

kiunet_model_a1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),    ## learning_rate is different from 3D
    loss=tf.keras.losses.BinaryCrossentropy(),  ##maybe categorical_crossentropy if labels one-hot encoded
    metrics=[keras.metrics.MeanIoU(num_classes=2), f1_metric]) #check metric, should be 4 classes

# Compile U-Net
unet_model_onehot = unet_from_code((128,128,3), 2)
m = tf.keras.metrics.MeanIoU(num_classes=256)

unet_model_onehot.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),    ## learning_rate is different from 3D
    loss=tf.keras.losses.BinaryCrossentropy(),  ##maybe categorical_crossentropy if labels one-hot encoded
    metrics=[keras.metrics.MeanIoU(num_classes=2), f1_metric])#check metric, should be 4 classes



# Train KiU-Net
## Instantiate logging for tensorboard
kiunet_train_log_dir = "logs/kiunet/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
kiunet_train_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=kiunet_train_log_dir, histogram_freq=1)


kiunet_model_a1.fit(x_train, y_onehot_train, batch_size=1, epochs=300, 
                 validation_data=(x_test, y_onehot_test), verbose=1,    ## validation_data = test_data, no early stopping
                 callbacks=[kiunet_train_tensorboard_callback])    ## log metrics in TensorBoard

# Save the entire model
## Save using SavedModel format
kiunet_model_a1.save('kiUnet_model_pb_3d_a')

## Save again using H5 format
kiunet_model_a1.save('kiUnet_model2_3d_a.h5', save_format='h5')



# Train U-Net
## Instaniate logging for TensorBoard
unet_train_log_dir = "logs/unet/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
unet_train_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=unet_train_log_dir, histogram_freq=1)

unet_model_onehot.fit(x_train, y_onehot_train, batch_size=1, epochs=300, 
                 validation_data=(x_test, y_onehot_test), verbose=1,    ## validation_data = test_data, no early stopping
                 callbacks=[unet_train_tensorboard_callback])    ## log metrics in TensorBoard

# Save trained U-Net
## Save using SavedModel format
unet_model_onehot.save('unet_model_pb')

## Save again using H5 format
unet_model_onehot.save('unet_model_h5.h5', save_format='h5')