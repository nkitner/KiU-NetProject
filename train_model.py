'''This file holds the training loops for the KiU-Net model found in model.py

Class: CISC867-Deep Learning
Author: Henry Lee
Date: 2021-10-07
'''
import tensorflow as tf
from tensorflow import keras
from model.py import model as kiunet
# insert import from data_processing #

kiunet.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),    ## learning_rate is different from 3D
    loss=keras.losses.binary_crossentropy(from_logits=False),
    metrics=[keras.metrics.F1Score(),    #check metric
    keras.metrics.MeanIoU(num_classes=   #unknown
)

kiunet.fit(x=x_train,
y=y_train,
batch_size=1,
epochs=400,
verbose='auto',
callbacks=keras.callbacks.TensorBoard(log_dir="./logs"),    ## log metrics in TensorBoard
validation_split=0.15,    ## uses 15% of training data as validation data
shuffle=True)