import tensorflow as tf
from constants import SQRT_EPS

def sqrt_safe(x):
    return tf.sqrt(tf.abs(x) + SQRT_EPS)

def acos_safe(x):
    return tf.math.acos(tf.clip_by_value(x, -1.0 + 1e-6, 1.0 - 1e-6))
