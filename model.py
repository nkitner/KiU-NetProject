# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# https://keras.io/examples/vision/oxford_pets_image_segmentation/

from tensorflow.keras import layers


def convolution_block_unet(input, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def convolution_block_kinet(input, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def encoder_block_unet(input, num_filters):
    x = convolution_block_unet(input, num_filters)
    # not sure abou this p
    p = layers.MaxPool2D((2, 2))(x)
    return x, p


def encoder_block_kinet(input, num_filters):
    x = convolution_block_kinet(input, num_filters)
    p = layers.MaxPool2D((2, 2))(x)
    return x, p


def decoder_block_unet(input, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = layers.Concatenate()([x, skip_features])
    x = convolution_block_unet(x, num_filters)
    return x


def decoder_block_kinet(input, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def crfb(x1, x2, num_filters, scale_factor):
    x2 = layers.Conv2D(num_filters, 3, padding="same")(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation("relu")(x2)
    x2 = layers.UpSampling2D(size=(scale_factor, scale_factor))(x2)
    out = layers.Concatenate()([x1, x2])
    return out


def crfb_kinet(unet, kinnet, scale_factor):
    return


def kiunet(input_shape):
    inputs = layers.Input(input_shape)

    # encoder block
    s1, p1 = encoder_block_unet(inputs, 16)  # s1 to be used at end
    k1, j1 = encoder_block_kinet(inputs, 16)  # k1 to be used at end

    residual_p1 = convolution_block_kinet(p1, 16)
    residual_k1 = convolution_block_unet(j1, 16)

    u1 = crfb(p1, residual_k1, 16, 0.25)
    o1 = crfb(j1, residual_p1, 16, 4)

    s2, p2 = encoder_block_unet(u1, 32)  # s2 to be used second to last
    k2, j2 = encoder_block_kinet(o1, 32)  # k2 to be used second to last

    residual_p2 = convolution_block_kinet(p2, 32)
    residual_k2 = convolution_block_unet(k2, 32)

    u2 = crfb(p2, residual_k2, 32, 0.0625)
    o2 = crfb(j2, residual_p2, 32, 16)

    s3, p3 = encoder_block_unet(p2, 64)
    k3, j3 = encoder_block_kinet(o1, 64)  # k3 to be used second to last

    residual_p3 = convolution_block_kinet(p3, 64)
    residual_k3 = convolution_block_unet(k3, 64)

    u3 = crfb(p2, residual_k2, 32, 0.015625)
    o3 = crfb(j2, residual_p2, 32, 64)

    # b1 = convolution_block_unet(p4, 256)

    # decoder block

    d1 = decoder_block_unet(b1, s4, 512)
    d2 = decoder_block_unet(d1, s3, 256)
    d3 = decoder_block_unet(d2, s2, 128)
    d4 = decoder_block_unet(d3, s1, 64)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model = keras.Model(inputs, outputs, name="U-Net")
    return model

    # ### [First half of the network: downsampling inputs] ###
    #
    # # Entry block
    # x = layers.Conv2D(16, 3, strides=1, padding="same")(inputs)
    # x = layers.MaxPool2D()(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation("relu")(x)
    #
    # x1 = layers.Conv2D(16, 3, strides=1, padding="same")(inputs)
    # x1 = tf.image.resize(x1, [32,32])
    # x1 = layers.BatchNormalization()(x1)
    # x1 = layers.Activation("relu")(x1)
    #
    # previous_block_activation = x  # Set aside residual
    #
    # # Blocks 1, 2, 3 are identical apart from the feature depth.
    # for filters in [64, 128, 256]:
    #     x = layers.Activation("relu")(x)
    #     x = layers.Conv2D(filters, 3, padding="same")(x)
    #     x = layers.MaxPool2D()(x)
    #
    #     y = layers.Activation("relu")(x)
    #     y = layers.SeparableConv2D(filters, 3, padding="same")(x)
    #     y = layers.BatchNormalization()(x)
    #
    #     # layers.Add used for cfrb
    #     y = layers.Add()([x,y])
    #     #
    #     # x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    #
    #     # Project residual
    #     residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
    #         previous_block_activation
    #     )
    #     x = layers.add([x, residual])  # Add back residual
    #     previous_block_activation = x  # Set aside next residual
    #
    # ### [Second half of the network: upsampling inputs] ###
    #
    # for filters in [256, 128, 64, 32]:
    #     x = layers.Activation("relu")(x)
    #     x = layers.Conv2D(filters, 3, padding="same")(x)
    #     x = layers.UpSampling2D()(x)
    #
    #     # x = layers.Activation("relu")(x)
    #     # x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
    #     # x = layers.BatchNormalization()(x)
    #
    #     # x = layers.UpSampling2D(2)(x)
    #
    #     # Project residual
    #     residual = layers.UpSampling2D()(previous_block_activation)
    #     residual = layers.Conv2D(filters, 1, padding="same")(residual)
    #     x = layers.add([x, residual])  # Add back residual
    #     previous_block_activation = x  # Set aside next residual
    #
    # # Add a per-pixel classification layer
    # outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    #
    # # Define the model
    # model = keras.Model(inputs, outputs)
    # return model


# def get_kitenet(img_size, num_classes):
#

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
img_size = (128, 128, 3)
num_classes = 3
model = kiunet(img_size)
model.summary()
