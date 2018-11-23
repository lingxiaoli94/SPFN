# differentiable total least square solver

import tensorflow as tf

def custom_svd_v_column(M, col_index=-1):
    # Must make sure M is finite. Otherwise cudaSolver might fail.
    assert_op = tf.Assert(tf.logical_not(tf.reduce_any(tf.logical_not(tf.is_finite(M)))), [M], summarize=10)
    with tf.control_dependencies([assert_op]):
        with tf.get_default_graph().gradient_override_map({'Svd': 'CustomSvd'}):
            s, u, v = tf.svd(M, name='Svd') # M = usv^T
            return v[:, :, col_index] 
        
def register_custom_svd_gradient():
    tf.RegisterGradient('CustomSvd')(custom_gradient_svd)

def custom_gradient_svd(op, grad_s, grad_u, grad_v):
    s, u, v = op.outputs
    # s - BxP
    # u - BxNxP, N >= P
    # v - BxPxP

    v_t = tf.transpose(v, [0, 2, 1])

    K = compute_svd_K(s)

    inner = tf.transpose(K, [0, 2, 1]) * tf.matmul(v_t, grad_v)
    inner = (inner + tf.transpose(inner, [0, 2, 1])) / 2

    # ignoring gradient coming from grad_s and grad_u for our purpose
    res = tf.matmul(u, tf.matmul(2 * tf.matmul(tf.matrix_diag(s), inner), v_t))

    return res

def compute_svd_K(s):
    # s should be BxP
    # res[b,i,j] = 1/(s[b,i]^2 - s[b,j]^2) if i != j, 0 otherwise
    # res will be BxPxP
    s = tf.square(s)
    res = tf.expand_dims(s, 2) - tf.expand_dims(s, 1)

    # making absolute value in res is at least 1e-10
    res = guard_one_over_matrix(res)

    return res

def guard_one_over_matrix(M, min_abs_value=1e-10):
    up = tf.matrix_band_part(tf.maximum(min_abs_value, M), 0, -1)
    low = tf.matrix_band_part(tf.minimum(-min_abs_value, M), -1, 0)
    M = up + low
    M += tf.eye(tf.shape(M)[1])
    M = 1 / M
    M -= tf.eye(tf.shape(M)[1])

    return M
    
def solve_weighted_tls(A, W):
    # A - BxNx3
    # W - BxN, positive weights
    # Find solution to min x^T A^T diag(W) A x = min ||\sqrt{diag(W)} A x||^2, subject to ||x|| = 1
    A_p = tf.expand_dims(A, axis=2) * tf.expand_dims(A, axis=3) # BxNx3x3
    W_p = tf.expand_dims(tf.expand_dims(W, axis=2), axis=3) # BxNx1x1
    M = tf.reduce_sum(W_p * A_p, axis=1) # Bx3x3
    x = custom_svd_v_column(M) # Bx3
    return x

