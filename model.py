# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _encoder_block_unet(input, num_filters):
    """
    Encoder logic for U-Net
    :param input: KerasTensor
    :param num_filters: Int - number of output filters in convolution
    :return: KerasTensor
    """
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.MaxPool2D((2, 2))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def _encoder_block_kinet(input, num_filters):
    """
    Encoder logic for Ki-Net
    :param input: KerasTensor
    :param num_filters: Int - number of output filters in convolution
    :return: KerasTensor
    """
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    p = layers.MaxPool2D((2, 2))(x)
    return x


def _decoder_block_unet(input, num_filters):
    """
    Decoder logic for U-Net
    :param input: KerasTensor
    :param num_filters: Int - number of output filters in convolution
    :return: KerasTensor
    """
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def _decoder_block_kinet(input, num_filters):
    """
    Decoder logic for Ki-net
    :param input: KerasTensor
    :param num_filters: Int - number of output filters in convolution
    :return: KerasTensor
    """
    x = layers.Conv2D(num_filters, 3, padding="same")(input)
    x = layers.MaxPool2D((2, 2))(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    print(type(x))
    return x


def _crfb(x1, x2, num_filters, scale_factor):
    """
    CRFB is the cross residual fusion block which fuses the outputs at the specified layers
    from one model to the other, and returns the results as input for the next step.
    :param x1: KerasTensor
    :param x2: KerasTensor
    :param num_filters: Int - number of output filters in convolution
    :param scale_factor: Float - scale factor for resizing (upsampling/downsampling) of output
    :return: KerasTensor
    """
    new_w = int(x2.shape[1] * scale_factor)
    new_h = int(x2.shape[2] * scale_factor)
    out = layers.Conv2D(num_filters, 3, padding="same")(x2)
    out = layers.Activation("relu")(out)
    out = tf.image.resize(out, [new_h, new_w])
    output = layers.Concatenate()([x1, out])
    return output


def kiunet(input_shape):
    """
    Architecture of the KiU-Net model, which combines the Ki-Net and U-Net architecture
    :param input_shape: Tuple of the shape of the input
    :return: keras.Model
    """
    inputs = layers.Input(shape=input_shape)

    # ENCODER BLOCK #

    s1 = _encoder_block_unet(inputs, 32)  # U NET ENCODER
    k1 = _encoder_block_kinet(inputs, 32)  # KINET ENCODER

    u1 = _crfb(s1, k1, 32, 0.25)  # CRFB U1 UNET
    o1 = _crfb(k1, s1, 32, 4)  # CRFB O1 KINET

    s2 = _encoder_block_unet(u1, 64)  # UNET ENCODER
    k2 = _encoder_block_kinet(o1, 64)  # KINET ENCODER

    u2 = _crfb(s2, k2, 64, 0.0625)  # CRFB U2 UNET
    o2 = _crfb(k2, s2, 64, 16)  # CRFB O2 KINET

    s3 = _encoder_block_unet(u2, 128)  # UNET ENCODER
    k3 = _encoder_block_kinet(o2, 128)  # KINET ENCODER

    u3 = _crfb(s3, k3, 128, 0.015625)  # CRFB U3 UNET
    o3 = _crfb(k3, s2, 128, 32)  # CRFB O3 KINET

    # DECODER BLOCK #

    d1_u = _decoder_block_unet(u3, 64)  # UNET DECODER
    d1_k = _decoder_block_kinet(o3, 64)  # KINET DECODER

    d_u1 = _crfb(d1_u, d1_k, 32, 0.0625)  # CRFB D_U1 UNET
    d_o1 = _crfb(d1_k, d1_u, 32, 16)  # CRFB D_O1 KINET

    out = layers.Concatenate()([d_u1, s2])  # CONCATENTATION D_U1 & S2 UNET
    out1 = layers.Concatenate()([d_o1, k2])  # CONCATENTATION D_O1 & K2 KINET

    d2_u = _decoder_block_unet(out, 32)  # UNET DECODER
    d2_k = _decoder_block_kinet(out1, 32)  # KINET DECODER

    d_u2 = _crfb(d2_u, d2_k, 16, 0.25)  # CRFB D_U2 UNET
    d_o2 = _crfb(d2_k, d2_u, 16, 4)  # CRFB D_O2 KINET

    out = layers.Concatenate()([d_u2, s1])  # CONCATENATION D_U2 & S1 UNET
    out1 = layers.Concatenate()([d_o2, k1])  # CONCATENATION D_O2 & K2 KINET

    d3_u = _decoder_block_unet(out, 16)  # UNET DECODER
    d3_k = _decoder_block_kinet(out1, 16)  # KINET DECODER

    out = layers.Concatenate()([d3_u, d3_k])  # FINAL CONCATENATION OUTPUT FROM UNET AND KINET

    out = layers.Conv2D(3, 3, padding="same", activation="relu")(out)  # FINAL CONVOLUTIONAL LAYER
    kiunet_model = keras.Model(inputs, out, name="U-Net")  # MODEL

    return kiunet_model


# Below code used for debugging

# Free up RAM in case the model definition cells were run multiple times
# keras.backend.clear_session()
#
# # Build model
# img_size = (128, 128, 3)
# num_classes = 3
# model = kiunet(img_size)
# model.summary()
