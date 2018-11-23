import os, sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '..', '..'))
sys.path.append(os.path.join(BASE_DIR, '..', '..', '..'))

from utils.tf_wrapper import batched_gather
from utils.geometry_utils import guarded_matrix_solve_ls, weighted_plane_fitting
from utils.tf_numerical_safe import sqrt_safe, acos_safe
from fitters.adaptors import *
from primitives import Cone

import tensorflow as tf
import numpy as np
from math import pi as PI

class ConeFitter:
    def primitive_name():
        return 'cone'

    def insert_prediction_placeholders(pred_ph, n_max_instances):
        pred_ph['cone_axis'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances, 3])
        pred_ph['cone_apex'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances, 3])
        pred_ph['cone_half_angle'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances])

    def normalize_parameters(parameters):
        parameters['cone_axis'] = tf.nn.l2_normalize(parameters['cone_axis'], axis=2)
        parameters['cone_half_angle'] = tf.clip_by_value(parameters['cone_half_angle'], 1e-4, PI / 2 - 1e-4)

    def insert_gt_placeholders(parameters_gt, n_max_instances):
        parameters_gt['cone_axis'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances, 3])

    def fill_gt_placeholders(feed_dict, parameters_gt, batch):
        feed_dict[parameters_gt['cone_axis']] = batch['cone_axis_gt']
        
    def compute_parameters(feed_dict, parameters):
        P = feed_dict['P']
        W = feed_dict['W']
        X = feed_dict['normal_per_point']
        batch_size = tf.shape(P)[0]
        n_points = tf.shape(P)[1]
        n_max_instances = W.get_shape()[2]
        W_reshaped = tf.reshape(tf.transpose(W, [0, 2, 1]), [batch_size * n_max_instances, n_points]) # BKxN
        # A - BKxNx3
        A = tf.reshape(tf.tile(tf.expand_dims(X, axis=1), [1, n_max_instances, 1, 1]), [batch_size * n_max_instances, n_points, 3]) # BKxNx3
        # b - BKxNx1
        b = tf.expand_dims(tf.reshape(tf.tile(tf.expand_dims(tf.reduce_sum(P * X, axis=2), axis=1), [1, n_max_instances, 1]), [batch_size * n_max_instances, n_points]), axis=2)
        
        apex = tf.reshape(guarded_matrix_solve_ls(A, b, W_reshaped), [batch_size, n_max_instances, 3]) # BxKx3
        X_tiled = A
        # TODO: use P-apex instead of X for plane fitting
        plane_n, plane_c = weighted_plane_fitting(X_tiled, W_reshaped)
        axis = tf.reshape(plane_n, [batch_size, n_max_instances, 3]) # BxKx3
        P_minus_apex_normalized = tf.nn.l2_normalize(tf.expand_dims(P, axis=2) - tf.expand_dims(apex, 1), axis=3) # BxNxKx3
        P_minus_apex_normalized_dot_axis = tf.reduce_sum(tf.expand_dims(axis, axis=1) * P_minus_apex_normalized, axis=3) # BxNxK
        # flip direction of axis if wrong
        sgn_axis = tf.sign(tf.reduce_sum(W * P_minus_apex_normalized_dot_axis, axis=1)) # BxK
        sgn_axis += tf.to_float(tf.equal(sgn_axis, 0.0)) # prevent sgn == 0
        axis *= tf.expand_dims(sgn_axis, axis=2) # BxKx3

        tmp = W * acos_safe(tf.abs(P_minus_apex_normalized_dot_axis)) # BxNxK
        W_sum = tf.reduce_sum(W, axis=1) # BxK
        half_angle = tf.reduce_sum(tmp, axis=1) / W_sum # BxK
        tf.clip_by_value(half_angle, 1e-3, PI / 2 - 1e-3) # angle cannot be too weird

        parameters['cone_apex'] = apex
        parameters['cone_axis'] = axis
        parameters['cone_half_angle'] = half_angle

    def compute_residue_loss(parameters, P_gt, matching_indices):
        return ConeFitter.compute_residue_single(
            *adaptor_matching([parameters['cone_apex'], parameters['cone_axis'], parameters['cone_half_angle']], matching_indices), 
            P_gt
        )

    def compute_residue_loss_pairwise(parameters, P_gt):
        return ConeFitter.compute_residue_single(
            *adaptor_pairwise([parameters['cone_apex'], parameters['cone_axis'], parameters['cone_half_angle']]), 
            adaptor_P_gt_pairwise(P_gt)
        )

    def compute_residue_single(apex, axis, half_angle, p):
        # apex: ...x3, axis: ...x3, half_angle: ..., p: ...x3
        v = p - apex
        v_normalized = tf.nn.l2_normalize(v, axis=-1)
        alpha = acos_safe(tf.reduce_sum(v_normalized * axis, axis=-1))
        return tf.square(tf.sin(tf.minimum(tf.abs(alpha - half_angle), PI / 2))) * tf.reduce_sum(v * v, axis=-1)

    def compute_parameter_loss(parameters_pred, parameters_gt, matching_indices, angle_diff):
        axis = batched_gather(parameters_pred['cone_axis'], matching_indices, axis=1)
        dot_abs = tf.abs(tf.reduce_sum(axis * parameters_gt['cone_axis'], axis=2))
        if angle_diff:
            return acos_safe(dot_abs) # BxK
        else:
            return 1.0 - dot_abs # BxK

    def extract_parameter_data_as_dict(primitives, n_max_instances):
        axis_gt = np.zeros(dtype=float, shape=[n_max_instances, 3])
        apex_gt = np.zeros(dtype=float, shape=[n_max_instances, 3])
        half_angle_gt = np.zeros(dtype=float, shape=[n_max_instances])
        for i, primitive in enumerate(primitives):
            if isinstance(primitive, Cone):
                axis_gt[i] = primitive.axis
                apex_gt[i] = primitive.apex
                half_angle_gt[i] = primitive.half_angle
        return {
            'cone_axis_gt': axis_gt,
        }

    def extract_predicted_parameters_as_json(fetched, k):
        cone = Cone(fetched['cone_apex'][k], fetched['cone_axis'][k], fetched['cone_half_angle'][k], z_min=0.0, z_max=5.0)
        return {
            'type': 'cone',
            'apex_x': cone.apex[0],
            'apex_y': cone.apex[1],
            'apex_z': cone.apex[2],
            'axis_x': cone.axis[0],
            'axis_y': cone.axis[1],
            'axis_z': cone.axis[2],
            'angle': cone.half_angle * 2,
            'z_min': cone.z_min,
            'z_max': cone.z_max,
        }

    def create_primitive_from_dict(d):
        assert d['type'] == 'cone'
        apex = np.array([d['apex_x'], d['apex_y'], d['apex_z']], dtype=float)
        axis = np.array([d['axis_x'], d['axis_y'], d['axis_z']], dtype=float)
        half_angle = float(d['semi_angle'])
        return Cone(apex=apex, axis=axis, half_angle=half_angle)


