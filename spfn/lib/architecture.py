import os, sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

from pointnet_plusplus.architectures import build_pointnet2_seg, build_pointnet2_cls
from utils.tf_wrapper import batched_gather
import fitter_factory

import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment

def get_per_point_model(scope, P, n_max_instances, is_training, bn_decay):
    ''' 
        Inputs:
            - P: BxNx3 tensor, the input point cloud
            - K := n_max_instances
        Outputs: a dict, containing
            - W: BxNxK, segmentation instances, fractional
            - normal_per_point: BxNx3, normal per point
            - type_per_point: BxNxT, type per points. NOTE: this is before taking softmax!
            - parameters - a dict, each entry is a BxKx... tensor
    '''

    n_registered_primitives = fitter_factory.get_n_registered_primitives()
    with tf.variable_scope(scope):
        net_results = build_pointnet2_seg('est_net', X=P, out_dims=[n_max_instances, 3, n_registered_primitives], is_training=is_training, bn_decay=bn_decay)
        W, normal_per_point, type_per_point = net_results
    W = tf.nn.softmax(W, axis=2) # BxNxK
    normal_per_point = tf.nn.l2_normalize(normal_per_point, axis=2) # BxNx3

    fitter_feed = {
        'P': P,
        'W': W,
        'normal_per_point': normal_per_point,
    }
    parameters = {}
    for fitter_cls in fitter_factory.get_all_fitter_classes():
        fitter_cls.compute_parameters(fitter_feed, parameters)
    
    return {
        'W': W,
        'normal_per_point': normal_per_point,
        'type_per_point': type_per_point,
        'parameters': parameters,
    }

def get_direct_regression_model(scope, P, n_max_instances, gt_dict, is_training, bn_decay):
    ''' 
        Inputs:
            - P: BxNx3 tensor, the input point cloud
            - K := n_max_instances
            - gt_dict: ground truth dictionary, needed since we are also computing the loss
        Outputs: (pred_dict, total_loss), where pred_dict contains
            - W: BxNxK, segmentation instances, binary
            - normal_per_point: BxNx3, normal per point, except in DPPN we don't predict normal, so all normals are constant
            - type_per_point: BxNxT, type per points, binary
            - parameters - a dict, each entry is a BxKx... tensor
    '''
    n_registered_primitives = fitter_factory.get_n_registered_primitives()
    batch_size = tf.shape(P)[0]
    n_points = tf.shape(P)[1]
    
    param_pair_list = get_param_dims_pair_list(n_max_instances)
    flattened_param_dims = [pr[1][0] * pr[1][1] for pr in param_pair_list]
    reg_result = build_pointnet2_cls('direct_reg_net', point_cloud=P, out_dims=flattened_param_dims, is_training=is_training, bn_decay=bn_decay)
    parameters = {}
    for idx, cls_result in enumerate(reg_result):
        param_name, param_dim = param_pair_list[idx]
        if param_dim[1] == 1:
            parameters[param_name] = cls_result # BxK
        else:
            parameters[param_name] = tf.reshape(cls_result, [-1, *param_dim])
    # normalize quantities
    for fitter_cls in fitter_factory.get_all_fitter_classes():
        fitter_cls.normalize_parameters(parameters)

    residue_losses = []
    for fitter_cls in fitter_factory.get_all_fitter_classes():
        residue_per_point = fitter_cls.compute_residue_loss_pairwise(parameters, gt_dict['points_per_instance']) # BxKxKxN'
        residue_avg = tf.reduce_mean(residue_per_point, axis=3) # BxKxK
        # residue_avg[b, k1, k2] is roughly the distance between gt instance k1 and predicted instance k2
        residue_losses.append(residue_avg)
    residue_matrix = tf.stack(residue_losses, axis=3) # BxKxKxT
    residue_matrix_flattened = tf.reshape(residue_matrix, shape=[batch_size, n_max_instances, -1]) # BxKxKT
    n_instance_gt = tf.reduce_max(gt_dict['instance_per_point'], axis=1) + 1
    mask_gt = tf.sequence_mask(n_instance_gt, maxlen=n_max_instances)
    matching_indices = tf.stop_gradient(tf.py_func(hungarian_matching, [residue_matrix_flattened, n_instance_gt], Tout=tf.int32)) # BxK
    matching_matrix = tf.reshape(tf.one_hot(matching_indices, depth=n_max_instances*n_registered_primitives), [batch_size, n_max_instances, n_max_instances, n_registered_primitives]) # BxKxKxT
    # only 1 element in matching_matrix[b, k, ..., ...] is nonzero
    direct_loss = tf.reduce_sum(matching_matrix * residue_matrix, axis=[2, 3]) # BxK
    direct_loss = tf.reduce_sum(direct_loss, axis=1) / tf.to_float(n_instance_gt) # B

    # reorder parameters
    matching_instance_id = tf.argmax(tf.reduce_sum(matching_matrix, axis=3), axis=2, output_type=tf.int32) # BxK
    matching_instance_type = tf.argmax(tf.reduce_sum(matching_matrix, axis=2), axis=2, output_type=tf.int32) # BxK
    for param in parameters:
        parameters[param] = batched_gather(parameters[param], matching_instance_id, axis=1)
    # now kth instance has type matching_instance_type[b, k] with parameters[b, k,...]

    # next construct W: BxNxK
    residue_per_point_matrix = []
    identity_matching_indices = tf.tile(tf.expand_dims(tf.range(n_max_instances), axis=0), [batch_size, 1]) # BxK
    for fitter_cls in fitter_factory.get_all_fitter_classes():
        residue_per_point = fitter_cls.compute_residue_loss(parameters, tf.tile(tf.expand_dims(P, axis=1), [1, n_max_instances, 1, 1]), matching_indices=identity_matching_indices) # BxKxN
        residue_per_point_matrix.append(residue_per_point)
    residue_per_point_matrix = tf.stack(residue_per_point_matrix, axis=3) # BxKxNxT

    # dist(P[b,n], instance k) = residue_per_point_matrix[b, k, n, matching_instance_type[b, k]]
    indices_0 = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(batch_size), axis=1), axis=1), [1, n_points, n_max_instances]) # BxNxK
    indices_1 = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(n_max_instances), axis=0), axis=0), [batch_size, n_points, 1]) # BxNxK
    indices_2 = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(n_points), axis=0), axis=2), [batch_size, 1, n_max_instances]) # BxNxK
    indices_3 = tf.tile(tf.expand_dims(matching_instance_type, axis=1), [1, n_points, 1]) # BxNxK
    P_to_instance_dist = tf.gather_nd(residue_per_point_matrix, indices=tf.stack([indices_0, indices_1, indices_2, indices_3], axis=3)) # BxNxK
    instance_per_point = tf.argmin(P_to_instance_dist, axis=2, output_type=tf.int32) # BxN
    W = tf.one_hot(instance_per_point, depth=n_max_instances) # BxNxK

    type_per_point = batched_gather(matching_instance_type, instance_per_point, axis=1)
    # we do not predict normal
    normal_per_point = tf.tile(tf.expand_dims(tf.expand_dims(tf.constant([1, 0, 0], dtype=tf.float32), axis=0), axis=0), [batch_size, n_points, 1]) # BxNx3
    return {
        'W': W,
        'normal_per_point': normal_per_point,
        'type_per_point': tf.one_hot(type_per_point, depth=n_registered_primitives),
        'parameters': parameters,
    }, direct_loss


def get_param_dims_pair_list(n_instances_per_type):
    pred_ph = {}
    for fitter_cls in fitter_factory.get_all_fitter_classes():
        fitter_cls.insert_prediction_placeholders(pred_ph, n_instances_per_type)
    param_pair_list = []
    for key in pred_ph:
        ph = pred_ph[key]
        if len(ph.get_shape()) == 2:
            param_pair_list.append((key, [n_instances_per_type, 1]))
        else:
            param_pair_list.append((key, [n_instances_per_type, ph.get_shape()[2]]))
    return param_pair_list

def hungarian_matching(cost, n_instance_gt):
    # cost is BxNxM
    B, N, M = cost.shape
    matching_indices = np.zeros([B, N], dtype=np.int32)
    for b in range(B):
        # limit to first n_instance_gt[b]
        _, matching_indices[b, :n_instance_gt[b]] = linear_sum_assignment(cost[b, :n_instance_gt[b], :])
    return matching_indices
