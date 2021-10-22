import tensorflow as tf
from tensorflow import keras
from data.add_data import import_data_numpy

x_test = import_data_numpy("./data/resized/test/img")
y_test = import_data_numpy("./data/resized/test/labelcol")

reconstructed_model = keras.models.load_model("kiUnet_model_h5.h5")
reconstructed_model.evaluate(x=x_test, y=y_test, verbose=1)
