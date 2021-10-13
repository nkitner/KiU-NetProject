'''This file holds the data loading functions
for training the KiU-Net model found in model.py

Class: CISC867-Deep Learning
Author: Henry Lee
Date: 2021-10-07
'''

import tensorflow as tf
import os
from PIL import Image


def import_data(path):
    """
    Imports png images and returns as a list of tensors to be used in training
    :param path: path to directory where images are
    :return: list of Tensors
    """
    tensor_list = []
    for file in os.listdir(path):
        # Converts to RGB because the vessel images are black and white
        png_img = Image.open(os.path.join(path, file)).convert('RGB')
        # Encodes images as tensors and adds to the list
        tensor_list.append(tf.io.encode_png(png_img))
    # Returns list of tensors to be used in model
    return tensor_list


x_train = import_data("./training/images/processed_img")
y_train = import_data("./training/av/processed_labels")

x_test = import_data("./testing/images/processed_img")
y_test = import_data("./testing/av/processed_labels")
