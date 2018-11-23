import tensorflow as tf
import numpy as np

def _variable_on_gpu(name, shape, initializer, dtype=tf.float32):
    with tf.device("/gpu:0"):
        return tf.get_variable(name, shape, initializer=initializer, dtype=dtype)

def _batch_norm_simple(scope, inputs, is_training, bn_decay=0.9):
    # Setting updates_collections to None will automatically add dependencies to update moving mean/variance
    # data_format = 'NHWC' so that normalization occurs at the last dimension
    return tf.contrib.layers.batch_norm(inputs, center=True, scale=True, is_training=is_training, decay=bn_decay, scope=scope, data_format='NHWC')

def fully_connected(scope, inputs, num_outputs, is_training, bn_decay=None, activation_fn=tf.nn.relu):
    # Input: inputs is ...xN
    # Returns: ...x[num_outputs]

    with tf.variable_scope(scope):
        weights = _variable_on_gpu('weights', [inputs.get_shape()[-1].value, num_outputs], initializer=tf.contrib.layers.xavier_initializer())
        biases = _variable_on_gpu('biases', [num_outputs], initializer=tf.constant_initializer(0.0))

        outputs = tf.nn.bias_add(tf.tensordot(inputs, weights, axes=[[-1], [0]]), biases)
        if bn_decay is not None:
            outputs = _batch_norm_simple('bn', outputs, is_training, bn_decay)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs

def dropout(scope, inputs, is_training, keep_prob):
    with tf.variable_scope(scope):
        return tf.cond(is_training, lambda: tf.nn.dropout(inputs, keep_prob), lambda: inputs)

def batched_gather(data, indices, axis):
    # data - Bx...xKx..., axis is where dimension K is
    # indices - BxK
    # output[b, ..., k, ...] = in[b, ..., indices[b, k], ...]
    assert axis >= 1
    ndims = data.get_shape().ndims # allow dynamic rank
    if axis > 1:
        # tranpose data to BxKx...
        perm = np.arange(ndims)
        perm[axis] = 1
        perm[1] = axis
        data = tf.transpose(data, perm=perm)
    batch_size = tf.shape(data)[0]
    batch_nums = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(batch_size), axis=1), axis=2), multiples=[1, tf.shape(indices)[1], 1]) # BxKx1
    indices = tf.concat([batch_nums, tf.expand_dims(indices, axis=2)], axis=2) # BxKx2
    gathered_data = tf.gather_nd(data, indices=indices)
    if axis > 1:
        gathered_data = tf.transpose(gathered_data, perm=perm)
    return gathered_data
