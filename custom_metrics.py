'''This script defines f1 score (Dice Coefficient) for use in Keras
as an evaluation metric.

Name: Henry Lee
Date: 2021-11-03
'''
from keras import backend as K  # import backend for custom metric


def recall_metric(y_true, y_pred):
    """
    Recall metric to evaluate model
    :param y_true: Numpy array of ground truth labels
    :param y_pred: Numpy array of predicted label
    :return: Double
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_metric(y_true, y_pred):
    """
    Precision metric to evaluate model
    :param y_true: Numpy array of ground truth labels
    :param y_pred: Numpy array of predicted label
    :return: Double
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_metric(y_true, y_pred):
    """
    f1 metric to evaluate model
    :param y_true: Numpy array of ground truth labels
    :param y_pred: Numpy array of predicted label
    :return: Double
    """
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
