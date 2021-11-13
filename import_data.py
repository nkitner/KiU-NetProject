'''This file holds the data loading functions
for training the KiU-Net model found in model.py

Class: CISC867-Deep Learning
Author: Nicole Kitner
Date: 2021-11-13
'''

import tensorflow as tf
import os
import numpy as np
import SimpleITK as sitk

def import_data_numpy(path):
    """
    Imports png images and returns as a list of tensors to be used in training
    :param path: path to directory where images are
    :return: list of Tensors
    """
    tensor_list = []
    for file in sorted(os.listdir(path)):
        if file.endswith(".png") or file.endswith(".tif"):
            # Converts to RGB because the vessel images are black and white
            png_img = Image.open(os.path.join(path, file))
            np_arr = normalize(np.array(png_img.getdata()).reshape((128, 128,3)))
            # tensor = tf.convert_to_tensor(np_arr)
            # Encodes images as tensors and adds to the list
            tensor_list.append(np_arr)
        # Returns list of tensors to be used in model
    # tensor_list = np.asarray(tensor_list)
    return np.asarray(tensor_list)

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
            np_arr = normalize(np.array(png_img.getdata()).reshape((128, 128,1)))
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