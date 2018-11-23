import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from pointnet_util import pointnet_sa_module, pointnet_fp_module
import tensorflow as tf
import tf_util

def build_pointnet2_seg(scope, X, out_dims, is_training, bn_decay):
    with tf.variable_scope(scope):
        l0_xyz = tf.slice(X, [0,0,0], [-1,-1,3])
        l0_points = tf.slice(X, [0,0,3], [-1,-1,0])

        # Set Abstraction layers
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points,
                npoint=512, radius=0.2, nsample=64, mlp=[64,64,128],
                mlp2=None, group_all=False, is_training=is_training,
                bn_decay=bn_decay, scope='layer1')

        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points,
                npoint=128, radius=0.4, nsample=64, mlp=[128,128,256],
                mlp2=None, group_all=False, is_training=is_training,
                bn_decay=bn_decay, scope='layer2')

        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points,
                npoint=None, radius=None, nsample=None, mlp=[256,512,1024],
                mlp2=None, group_all=True, is_training=is_training,
                bn_decay=bn_decay, scope='layer3')

        # Feature Propagation layers
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points,
                [256,256], is_training, bn_decay, scope='fa_layer1')

        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points,
                [256,128], is_training, bn_decay, scope='fa_layer2')

        l0_points = pointnet_fp_module(l0_xyz, l1_xyz,
                tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128],
                is_training, bn_decay, scope='fa_layer3')

        # FC layers
        net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True,
                is_training=is_training, scope='fc1', bn_decay=bn_decay)

        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                scope='dp1')

        results = []
        for idx, out_dim in enumerate(out_dims):
            current_result = tf_util.conv1d(net, out_dim, 1, padding='VALID', activation_fn=None, scope='fc2_{}'.format(idx))
            results.append(current_result)

        return results

def build_pointnet2_cls(scope, point_cloud, out_dims, is_training, bn_decay):
    with tf.variable_scope(scope):
        batch_size = tf.shape(point_cloud)[0]
        l0_xyz = point_cloud
        l0_points = None

        # Set abstraction layers
        # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
        # So we only use NCHW for layer 1 until this issue can be resolved.
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True)
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

        # Fully connected layers
        net = tf.reshape(l3_points, [batch_size, 1024])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')

        results = []
        for idx, out_dim in enumerate(out_dims):
            current_result = tf_util.fully_connected(net, out_dim, activation_fn=None, scope='fc3_{}'.format(idx))
            results.append(current_result)

        return results
