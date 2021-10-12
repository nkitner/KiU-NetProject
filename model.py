# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# https://keras.io/examples/vision/oxford_pets_image_segmentation/

from tensorflow.keras import layers


def encoder_block_unet(input, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.MaxPool2D((2, 2))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def encoder_block_kinet(input, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    p = layers.MaxPool2D((2, 2))(x)
    return x, p


def decoder_block_unet(input, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def decoder_block_kinet(input, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.MaxPool2D((2, 2))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def crfb(x1, x2, num_filters, scale_factor):
    new_w = int(x2.shape[1] * scale_factor)
    new_h = int(x2.shape[2] * scale_factor)
    out = layers.Conv2D(num_filters, 3, padding="same")(x2)
    out = layers.Activation("relu")(out)
    out = tf.image.resize(out, [new_h, new_w])
    output = layers.Concatenate()([x1, out])
    return output


def kiunet(input_shape):
    inputs = layers.Input(input_shape)

    # encoder block
    s1, p1 = encoder_block_unet(inputs, 32)  # s1 to be used at end
    k1, j1 = encoder_block_kinet(inputs, 32)  # k1 to be used at end

    u1 = crfb(s1, k1, 32, 0.25)
    o1 = crfb(k1, s1, 32, 4)

    s2, p2 = encoder_block_unet(s1, 64)  # s2 to be used second to last
    k2, j2 = encoder_block_kinet(k1, 64)  # k2 to be used second to last

    u2 = crfb(s2, k2, 64, 0.0625)
    o2 = crfb(k2, s2, 64, 16)

    s3, p3 = encoder_block_unet(s2, 128)
    k3, j3 = encoder_block_kinet(k2, 128)  # k3 to be used second to last

    u3 = crfb(s3, k3, 128, 0.015625)
    o3 = crfb(k3, s2, 128, 32)

    # b1 = convolution_block_unet(p4, 256)

    # decoder block

    d1_u = decoder_block_unet(u3, 64)
    d1_k = decoder_block_kinet(o3, 64)
    out = crfb(d1_u, d1_k, 32, 0.0625)
    out1 = crfb(d1_k, d1_u, 32, 16)
    out = layers.Concatenate()([out, u2])
    out1 = layers.Concatenate()([out1, o2])

    d2_u = decoder_block_unet(out, 32)
    d2_k = decoder_block_kinet(out1, 32)
    out = crfb(d2_u, d2_k, 16, 0.25)
    out1 = crfb(d2_k, d2_u, 16, 4)
    out = layers.Concatenate()([out, u1])
    out1 = layers.Concatenate()([out1, o1])

    d3_u = decoder_block_unet(out, 16)
    d3_k = decoder_block_kinet(out1, 16)

    out = layers.Concatenate()([d3_u, d3_k])
    out = layers.Activation("relu")(out)

    out = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(out)
    kiunet_model = keras.Model(inputs, out, name="U-Net")
    return kiunet_model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
img_size = (128, 128, 3)
num_classes = 3
model = kiunet(img_size)
model.summary()
