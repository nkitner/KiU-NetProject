import sys

import tensorflow as tf
from tensorflow import keras
from data.add_data import import_data_numpy, import_data_numpy_mask
from sklearn.metrics import f1_score
import numpy as np
import custom_metrics as cm

x_train = import_data_numpy("./data/resized/train/img")
y_train = import_data_numpy_mask("./data/resized/train/labelcol")

x_test = import_data_numpy("./data/resized/test/img")
y_test = import_data_numpy_mask("./data/resized/test/labelcol")

y_train_binary = np.copy(y_train)
y_test_binary = np.copy(y_test)

for i in range(len(y_train)):
    y_train_binary[i][y_train_binary[i] > 0] = 1
    y_test_binary[i][y_test_binary[i] > 0] = 1

y_onehot_train = keras.utils.to_categorical(y_train_binary, 2)
y_onehot_test = keras.utils.to_categorical(y_test_binary, 2)


def evaluate_segnet():
    print("Evaluating...")
    segnet_model = keras.models.load_model("segnet_model.h5",  custom_objects={'f1_metric': cm.f1_metric})
    test_pred_segnet = segnet_model.predict(x_test)
    y_classes_segnet = test_pred_segnet.argmax(axis=-1).flatten()
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    y_onehot = y_onehot_test[:, :, :, 1].flatten()
    m.update_state(y_classes_segnet, y_onehot)
    segnet_dice = m.result().numpy()
    print("\nSeg-Net Evaluation:")
    print("Dice Score: {}".format(segnet_dice))
    segnet_jaccard = f1_score(y_classes_segnet, y_onehot, average="macro")
    print("Jaccard Score: {}".format(segnet_jaccard))


def evaluate_unet():
    print("Evaluating...")
    unet_model = keras.models.load_model("unet_model.h5", custom_objects={'f1_metric': cm.f1_metric})
    test_pred_unet = unet_model.predict(x_test)
    y_classes_unet = test_pred_unet.argmax(axis=-1).flatten()
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    y_onehot = y_onehot_test[:, :, :, 1].flatten()
    m.update_state(y_classes_unet, y_onehot)
    unet_dice = m.result().numpy()
    print("\nU-Net Evaluation:")
    print("Dice Score: {}".format(unet_dice))
    unet_jaccard = f1_score(y_classes_unet, y_onehot, average="macro")
    print("Jaccard Score: {}".format(unet_jaccard))


def evaluate_kiunet():
    print("Evaluating...")
    kiunet_model = keras.models.load_model("kiUnet_model.h5", custom_objects={'f1_metric': cm.f1_metric})
    test_pred_kiunet = kiunet_model.predict(x_test)
    y_classes_kiunet = test_pred_kiunet.argmax(axis=-1).flatten()
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    y_onehot = y_onehot_test[:, :, :, 1].flatten()
    m.update_state(y_classes_kiunet, y_onehot)
    kiunet_dice = m.result().numpy()
    print("\nKiU-Net Evaluation:")
    print("Dice Score: {}".format(kiunet_dice))
    kiunet_jaccard = f1_score(y_classes_kiunet, y_onehot, average="macro")
    print("Jaccard Score: {}".format(kiunet_jaccard))


def evaluate_all():
    evaluate_segnet()
    evaluate_unet()
    evaluate_kiunet()


if __name__ == '__main__':
    args = sys.argv
    globals()[sys.argv[1]]()
