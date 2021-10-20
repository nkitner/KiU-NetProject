import tensorflow as tf

# GRAPH_PB_PATH = 'model/'

import tensorflow as tf
from tensorflow.python.platform import gfile

import tensorflow as tf
import data.add_data as data
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
import time
# import keras
import pydot

from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

#
# with tf.compat.v1.Session() as sess:
#     model_filename = 'model/saved_model.pb'
#     with gfile.FastGFile(model_filename, 'rb') as f:
#         data = compat.as_bytes(f.read())
#         sm = saved_model_pb2.SavedModel()
#         sm.ParseFromString(data)
#         g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
x_train = data.import_data_numpy("./data/training/images/processed_labels")
y_train = data.import_data_numpy("./data/training/av/processed_labels")

x_test = data.import_data_numpy("./data/testing/images/processed_labels")
y_test = data.import_data_numpy("./data/testing/av/processed_labels")

NAME = "RITE_cnn-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='log/{}'.format(NAME))

model = tf.saved_model.load("kiUnet_model")
model.summary()