import os, sys
from os.path import dirname
sys.path.append(dirname(__file__))

import tensorflow as tf
from differentiable_tls import solve_weighted_tls
from constants import SQRT_EPS, DIVISION_EPS, LS_L2_REGULARIZER

def compute_consistent_plane_frame(normal):
    # Input:  normal is Bx3
    # Returns: x_axis, y_axis, both of dimension Bx3
    batch_size = tf.shape(normal)[0]
    candidate_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # Actually, 2 should be enough. This may still cause singularity TODO!!!
    y_axes = []
    for tmp_axis in candidate_axes:
        tf_axis = tf.tile(tf.expand_dims(tf.constant(dtype=tf.float32, value=tmp_axis), axis=0), [batch_size, 1]) # Bx3
        y_axes.append(tf.cross(normal, tf_axis))
    y_axes = tf.stack(y_axes, axis=0) # QxBx3
    y_axes_norm = tf.norm(y_axes, axis=2) # QxB
    # choose the axis with largest norm
    y_axes_chosen_idx = tf.argmax(y_axes_norm, axis=0) # B
    # y_axes_chosen[b, :] = y_axes[y_axes_chosen_idx[b], b, :]
    indices_0 = tf.tile(tf.expand_dims(y_axes_chosen_idx, axis=1), [1, 3]) # Bx3
    indices_1 = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, 3]) # Bx3
    indices_2 = tf.tile(tf.expand_dims(tf.range(3), axis=0), [batch_size, 1]) # Bx3
    indices = tf.stack([tf.cast(indices_0, tf.int32), indices_1, indices_2], axis=2) # Bx3x3
    y_axes = tf.gather_nd(y_axes, indices=indices) # Bx3
    if tf.VERSION == '1.4.1':
        y_axes = tf.nn.l2_normalize(y_axes, dim=1)
    else:
        y_axes = tf.nn.l2_normalize(y_axes, axis=1)
    x_axes = tf.cross(y_axes, normal) # Bx3

    return x_axes, y_axes

def weighted_plane_fitting(P, W):
    # P - BxNx3
    # W - BxN
    # Returns n, c, with n - Bx3, c - B
    WP = P * tf.expand_dims(W, axis=2) # BxNx3
    W_sum = tf.reduce_sum(W, axis=1) # B
    P_weighted_mean = tf.reduce_sum(WP, axis=1) / tf.maximum(tf.expand_dims(W_sum, 1), DIVISION_EPS) # Bx3
    A = P - tf.expand_dims(P_weighted_mean, axis=1) # BxNx3
    n = solve_weighted_tls(A, W) # Bx3
    c = tf.reduce_sum(n * P_weighted_mean, axis=1)
    return n, c

'''
    L = \sum_{i=1}^m w_i [(p_i - x)^2 - r^2]^2
    dL/dr = 0 => r^2 = \frac{1}{\sum_i w_i} \sum_j w_j (p_j - x)^2
    => L = \sum_{i=1}^m w_i[p_i^2 - \frac{\sum_j w_j p_j^2}{\sum_j w_j} + 2x \cdot (-p_i + \frac{\sum_j w_j p_j}{\sum_j w_j})]^2
    So
    A_i = 2\sqrt{w_i} (\frac{w_j p_j}{\sum_j w_j} - p_i)
    b_i = \sqrt{w_i}[\frac{\sum_j w_j p_j^2}{\sum_j w_j} - p_i^2]
    So \argmin_x ||Ax-b||^2 gives the best center of the sphere
'''
def weighted_sphere_fitting(P, W):
    # P - BxNxD
    # W - BxN
    W_sum = tf.reduce_sum(W, axis=1) # B
    WP_sqr_sum = tf.reduce_sum(W * tf.reduce_sum(tf.square(P), axis=2), axis=1) # B
    P_sqr = tf.reduce_sum(tf.square(P), axis=2) # BxN
    b = tf.expand_dims(tf.expand_dims(WP_sqr_sum / tf.maximum(W_sum, DIVISION_EPS), axis=1) - P_sqr, axis=2) # BxNx1
    WP_sum = tf.reduce_sum(tf.expand_dims(W, axis=2) * P, axis=1) # BxD
    A = 2 * (tf.expand_dims(WP_sum / tf.expand_dims(tf.maximum(W_sum, DIVISION_EPS), axis=1), axis=1) - P) # BxNxD

    # Seek least norm solution to the least square
    center = guarded_matrix_solve_ls(A, b, W) # BxD
    W_P_minus_C_sqr_sum = P - tf.expand_dims(center, axis=1) # BxNxD
    W_P_minus_C_sqr_sum = W * tf.reduce_sum(tf.square(W_P_minus_C_sqr_sum), axis=2) # BxN
    r_sqr = tf.reduce_sum(W_P_minus_C_sqr_sum, axis=1) / tf.maximum(W_sum, DIVISION_EPS) # B

    return {'center': center, 'radius_squared': r_sqr}

def guarded_matrix_solve_ls(A, b, W, condition_number_cap=1e5):
    # Solve weighted least square ||\sqrt(W)(Ax-b)||^2
    # A - BxNxD
    # b - BxNx1
    # W - BxN

    sqrt_W = tf.sqrt(tf.maximum(W, SQRT_EPS)) # BxN
    A *= tf.expand_dims(sqrt_W, axis=2) # BxNxD
    b *= tf.expand_dims(sqrt_W, axis=2) # BxNx1
    # Compute singular value, trivializing the problem when condition number is too large
    AtA = tf.matmul(a=A, b=A, transpose_a=True)
    s, _, _ = [tf.stop_gradient(u) for u in tf.svd(AtA)] # s will be BxD
    mask = tf.less(s[:, 0] / s[:, -1], condition_number_cap) # B
    A *= tf.to_float(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=2)) # zero out badly conditioned data
    x = tf.matrix_solve_ls(A, b, l2_regularizer=LS_L2_REGULARIZER, fast=True) # BxDx1 
    return tf.squeeze(x, axis=2) # BxD

