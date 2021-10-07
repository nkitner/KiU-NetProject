# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# https://keras.io/examples/vision/oxford_pets_image_segmentation/
# this is taken from above link, will be changed but currently being used as practice

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers


def get_unet(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(2, 3, strides=1, padding="same")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 32, 16]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.MaxPool2D()(x)

        # x = layers.Activation("relu")(x)
        # x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        # x = layers.BatchNormalization()(x)
        #
        # x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [16, 32, 64]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.UpSampling2D()(x)

        # x = layers.Activation("relu")(x)
        # x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        # x = layers.BatchNormalization()(x)

        # x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
img_size=(128,128)
num_classes=3
model = get_unet(img_size, num_classes)
model.summary()
