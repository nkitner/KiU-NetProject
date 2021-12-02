'''This file holds the data loading functions
for training the KiU-Net model found in model.py

Class: CISC867-Deep Learning
Author: Henry Lee
Date: 2021-10-07
'''

import tensorflow as tf
import os
from PIL import Image
import numpy as np


def import_data_numpy(path):
    """
    Imports png images and returns as a list of tensors to be used in training
    :param path: path to directory where images are
    :return: list of Tensors
    """
    tensor_list = []
    for file in os.listdir(path):
        if file.endswith(".png") or file.endswith(".tif"):
            png_img = Image.open(os.path.join(path, file))
            # Encodes images as numpy arrays and adds to the list
            tensor_list.append(np.array(png_img.getdata()).reshape((128, 128, 3)))
        # Returns list of numpy arrays to be used in model
    tensor_list = np.asarray(tensor_list)
    return tensor_list


def import_data_numpy_mask(path):
    """
    Imports png images and returns as a list of tensors to be used in training
    :param path: path to directory where images are
    :return: list of Tensors
    """
    tensor_list = []
    for file in sorted(os.listdir(path)):
        if file.endswith(".png") or file.endswith(".tif"):
            # Converts to RGB because the vessel images are black and white
            png_img = Image.open(os.path.join(path, file)).convert('L')
            np_arr = normalize(np.array(png_img.getdata()).reshape((128, 128, 1)))
            # tensor = tf.convert_to_tensor(np_arr)
            # Encodes images as tensors and adds to the list
            tensor_list.append(np_arr)
        # Returns list of tensors to be used in model
    # tensor_list = np.asarray(tensor_list)
    return np.asarray(tensor_list)


def normalize(input_image):
    max = np.max(input_image)
    input_image = tf.cast(input_image, tf.float32) / max
    return input_image

# x_train = import_data_numpy("./training/images/processed_img")
# y_train = import_data_numpy("./training/av/processed_labels")
#
# x_test = import_data_numpy("./testing/images/processed_img")
# y_test = import_data_numpy("./testing/av/processed_labels")
