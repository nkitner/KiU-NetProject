'''This file holds the data loading functions
for training the KiU-Net model found in model.py

Class: CISC867-Deep Learning
Author: Henry Lee
Date: 2021-10-07
'''

import tensorflow as tf

def import_data(path):
    '''Imports data from retina dataset
    '''
    for file in path:
        #how do we import images into a tf tensor...?


# fxn below should be removed I think #
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

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)