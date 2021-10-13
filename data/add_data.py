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


y_train = import_data("./training/av")
# vessel_training_tensor = import_data("./training/vessel")
x_train = import_data("./training/images")

x_test = import_data("./testing/av")
# vessel_testing_tensor = import_data("./testing/vessel")
y_test = import_data("./testing/images")

# train_input_fn = make_input_fn(dftrain, y_train)
# eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
