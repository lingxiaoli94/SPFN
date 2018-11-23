import os, sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '..', '..'))
sys.path.append(os.path.join(BASE_DIR, '..', '..', '..'))

from utils.tf_wrapper import batched_gather
from utils.differentiable_tls import solve_weighted_tls
from utils.geometry_utils import compute_consistent_plane_frame, weighted_sphere_fitting
from utils.tf_numerical_safe import sqrt_safe, acos_safe
from fitters.adaptors import *
from primitives import Cylinder

import tensorflow as tf
import numpy as np

class CylinderFitter:
    def primitive_name():
        return 'cylinder'

    def insert_prediction_placeholders(pred_ph, n_max_instances):
        pred_ph['cylinder_axis'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances, 3])
        pred_ph['cylinder_center'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances, 3])
        pred_ph['cylinder_radius_squared'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances])

    def normalize_parameters(parameters):
        parameters['cylinder_axis'] = tf.nn.l2_normalize(parameters['cylinder_axis'], axis=2)
        parameters['cylinder_radius_squared'] = tf.clip_by_value(parameters['cylinder_radius_squared'], 1e-4, 1e6)

    def insert_gt_placeholders(parameters_gt, n_max_instances):
        parameters_gt['cylinder_axis'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances, 3])

    def fill_gt_placeholders(feed_dict, parameters_gt, batch):
        feed_dict[parameters_gt['cylinder_axis']] = batch['cylinder_axis_gt']

    def compute_parameters(feed_dict, parameters):
        P = feed_dict['P']
        W = feed_dict['W']
        X = feed_dict['normal_per_point']
        # First determine n as the solution to \min \sum W_i (X_i \cdot n)^2
        batch_size = tf.shape(P)[0]
        n_points = tf.shape(P)[1]
        n_max_primitives = tf.shape(W)[2]
        W_reshaped = tf.reshape(tf.transpose(W, [0, 2, 1]), [batch_size * n_max_primitives, n_points]) # BKxN
        X_reshaped = tf.reshape(tf.tile(tf.expand_dims(X, axis=1), [1, n_max_primitives, 1, 1]), [batch_size * n_max_primitives, n_points, 3]) # BKxNx3
        n = tf.reshape(solve_weighted_tls(X_reshaped, W_reshaped), [batch_size, n_max_primitives, 3]) # BxKx3

        x_axes, y_axes = compute_consistent_plane_frame(tf.reshape(n, [batch_size * n_max_primitives, 3]))
        x_axes = tf.reshape(x_axes, [batch_size, n_max_primitives, 3]) # BxKx3
        y_axes = tf.reshape(y_axes, [batch_size, n_max_primitives, 3]) # BxKx3

        x_coord = tf.reduce_sum(tf.expand_dims(P, axis=1) * tf.expand_dims(x_axes, axis=2), axis=3) # BxKxN
        y_coord = tf.reduce_sum(tf.expand_dims(P, axis=1) * tf.expand_dims(y_axes, axis=2), axis=3) # BxKxN
        P_proj = tf.stack([x_coord, y_coord], axis=3) # BxKxNx2, 2D projection point
        P_proj_reshaped = tf.reshape(P_proj, [batch_size * n_max_primitives, n_points, 2]) # BKxNx2
        circle_fitting_result = weighted_sphere_fitting(P_proj_reshaped, W_reshaped)

        circle_center = tf.reshape(circle_fitting_result['center'], [batch_size, n_max_primitives, 2]) # BxKx2
        c = tf.expand_dims(circle_center[:, :, 0], axis=2) * x_axes + tf.expand_dims(circle_center[:, :, 1], axis=2) * y_axes # BxKx3
        r_sqr = tf.reshape(circle_fitting_result['radius_squared'], [batch_size, n_max_primitives]) # BxK

        parameters['cylinder_axis'] = n
        parameters['cylinder_center'] = c
        parameters['cylinder_radius_squared'] = r_sqr

    def compute_residue_loss(parameters, P_gt, matching_indices):
        return CylinderFitter.compute_residue_single(
            *adaptor_matching([parameters['cylinder_axis'], parameters['cylinder_center'], parameters['cylinder_radius_squared']], matching_indices), 
            P_gt
        )

    def compute_residue_loss_pairwise(parameters, P_gt):
        return CylinderFitter.compute_residue_single(
            *adaptor_pairwise([parameters['cylinder_axis'], parameters['cylinder_center'], parameters['cylinder_radius_squared']]), 
            adaptor_P_gt_pairwise(P_gt)
        )

    def compute_residue_single(axis, center, radius_squared, p):
        p_minus_c = p - center
        p_minus_c_sqr = tf.reduce_sum(tf.square(p_minus_c), axis=-1)
        p_minus_c_dot_n = tf.reduce_sum(p_minus_c * axis, axis=-1)
        return tf.square(sqrt_safe(p_minus_c_sqr - tf.square(p_minus_c_dot_n)) - sqrt_safe(radius_squared))

    def compute_parameter_loss(parameters_pred, parameters_gt, matching_indices, angle_diff):
        n = batched_gather(parameters_pred['cylinder_axis'], matching_indices, axis=1)
        dot_abs = tf.abs(tf.reduce_sum(n * parameters_gt['cylinder_axis'], axis=2))
        if angle_diff:
            return acos_safe(dot_abs) # BxK
        else:
            return 1.0 - dot_abs # BxK

    def extract_parameter_data_as_dict(primitives, n_max_primitives):
        n = np.zeros(dtype=float, shape=[n_max_primitives, 3])
        for i, primitive in enumerate(primitives):
            if isinstance(primitive, Cylinder):
                n[i] = primitive.axis
        return {
            'cylinder_axis_gt': n
        }

    def extract_predicted_parameters_as_json(fetched, k):
        cylinder = Cylinder(fetched['cylinder_center'][k], np.sqrt(fetched['cylinder_radius_squared'][k]), fetched['cylinder_axis'][k], height=5)
        return {
            'type': 'cylinder',
            'center_x': cylinder.center[0],
            'center_y': cylinder.center[1],
            'center_z': cylinder.center[2],
            'radius': cylinder.radius,
            'axis_x': cylinder.axis[0],
            'axis_y': cylinder.axis[1],
            'axis_z': cylinder.axis[2],
            'height': cylinder.height,
        }

    def create_primitive_from_dict(d):
        assert d['type'] == 'cylinder'
        location = np.array([d['location_x'], d['location_y'], d['location_z']], dtype=float)
        axis = np.array([d['axis_x'], d['axis_y'], d['axis_z']], dtype=float)
        radius = float(d['radius'])
        return Cylinder(center=location, radius=radius, axis=axis)

