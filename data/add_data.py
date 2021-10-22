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


# x_train = import_data_numpy("./training/images/processed_img")
# y_train = import_data_numpy("./training/av/processed_labels")
#
# x_test = import_data_numpy("./testing/images/processed_img")
# y_test = import_data_numpy("./testing/av/processed_labels")
