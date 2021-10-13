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


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=1):
    '''Generates an input_function instance

    Args:
        1) data_df(df): pandas dataframe holding input data
        2) label_df(df): pandas dataframe holding ground truth labels for the input data
        3) num_epochs(int): epochs to train for
        4) shuffle(bool): toggles randomization of input data
        5) batch_size(int): batch size
    '''

    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df, label_df)))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)

    return input_function


av_training_tensor = import_data("./training/av")
vessel_training_tensor = import_data("./training/vessel")
image_training_tensor = import_data("./training/images")

av_testing_tensor = import_data("./testing/av")
vessel_testing_tensor = import_data("./testing/vessel")
image_testing_tensor = import_data("./testing/images")

# train_input_fn = make_input_fn(dftrain, y_train)
# eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
