import keras.backend as K
import tensorflow as tf


def error_to_signal(y_true, y_pred):
    return K.sum(tf.pow(y_true - y_pred, 2), axis=0) / (K.sum(tf.pow(y_true, 2), axis=0) + 1e-10)
