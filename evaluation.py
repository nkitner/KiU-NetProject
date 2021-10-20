from datetime import time

import tensorflow as tf
from keras.metrics import MeanIoU
from tensorflow.metrics import F1Score
from keras.metrics import AUC
from tensorflow.keras.callbacks import TensorBoard

# before starting:
# 0a. predict testing images
# 0b. save newly predicted test images
# 0c. read ground truth image

# y_true =
# y_pred =
# set the possible number of classes the prediction task can have
# num_class =

# define mIOU
from tensorflow.python.ops.metrics_impl import mean_iou

mIOU = mean_iou(num_classes=num_class)
# update mIOU
mIOU.update_state(y_true, y_pred)
mIOU_result = mIOU.result()
print('mIOU:', mIOU_result.numpy())

# define F1 score
dice = F1Score(num_classes=num_class)
# update f1 socre
dice.update_state(y_true, y_pred)
dice_result = dice.result()
print('F1 Score:', dice_result.numpy())

# define auc
auc = AUC(num_thresholds=200) #'num_thresholds' controls the degree of discretization with larger numbers of thresholds more closely approximating the true AUC. Defaults to 200.
# update auc
auc.update_state(y_true, y_pred)
auc_result = auc.result()
print('AUC:',auc_result.numpy())


# Visualize evaluation of both training and testing using Tensorboard- Accuracy, Loss
NAME = "RITE_cnn-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='log/{}'.format(NAME))

model.fit(............., callback=[tensorboard])

# hihihi