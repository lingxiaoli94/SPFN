import tensorflow as tf
from utils.tf_wrapper import batched_gather

def adaptor_matching(param_list, matching_indices):
    return [tf.expand_dims(batched_gather(param, matching_indices, axis=1), axis=2) for param in param_list]

def adaptor_pairwise(param_list):
    return [tf.expand_dims(tf.expand_dims(param, axis=1), axis=3) for param in param_list]

def adaptor_P_gt_pairwise(P_gt):
    # P_gt is BxKxN'x3, making it BxKxKxN'x3
    # return tf.tile(tf.expand_dims(P_gt, axis=2), [1, 1, tf.shape(P_gt)[1], 1, 1])
    return tf.expand_dims(P_gt, axis=2)
