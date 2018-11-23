import os, sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '..', '..'))
sys.path.append(os.path.join(BASE_DIR, '..', '..', '..'))

from utils.tf_wrapper import batched_gather
from utils.geometry_utils import weighted_plane_fitting
from utils.tf_numerical_safe import acos_safe
from fitters.adaptors import *
from primitives import Plane

import tensorflow as tf
import numpy as np

''' Fitters should have no knowledge of ground truth labels,
    and should assume all primitives are of the same type.
    They should predict a primitive for every column.
'''
class PlaneFitter:
    def primitive_name():
        return 'plane'

    def insert_prediction_placeholders(pred_ph, n_max_instances):
        pred_ph['plane_n'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances, 3])
        pred_ph['plane_c'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances])

    def normalize_parameters(parameters):
        parameters['plane_n'] = tf.nn.l2_normalize(parameters['plane_n'], axis=2)

    def insert_gt_placeholders(parameters_gt, n_max_instances):
        parameters_gt['plane_n'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances, 3])

    def fill_gt_placeholders(feed_dict, parameters_gt, batch):
        feed_dict[parameters_gt['plane_n']] = batch['plane_n_gt']

    def compute_parameters(feed_dict, parameters):
        P = feed_dict['P']
        W = feed_dict['W']
        batch_size = tf.shape(P)[0]
        n_points = tf.shape(P)[1]
        n_max_instances = tf.shape(W)[2]
        W_reshaped = tf.reshape(tf.transpose(W, [0, 2, 1]), [batch_size * n_max_instances, n_points]) # BKxN
        P_tiled = tf.reshape(tf.tile(tf.expand_dims(P, axis=1), [1, n_max_instances, 1, 1]), [batch_size * n_max_instances, n_points, 3]) # BKxNx3, important there to match indices in W_reshaped!!!
        n, c = weighted_plane_fitting(P_tiled, W_reshaped) # BKx3
        parameters['plane_n'] = tf.reshape(n, [batch_size, n_max_instances, 3]) # BxKx3
        parameters['plane_c'] = tf.reshape(c, [batch_size, n_max_instances]) # BxK

    def compute_residue_loss(parameters, P_gt, matching_indices):
        return PlaneFitter.compute_residue_single(
            *adaptor_matching([parameters['plane_n'], parameters['plane_c']], matching_indices), 
            P_gt
        )

    def compute_residue_loss_pairwise(parameters, P_gt):
        return PlaneFitter.compute_residue_single(
            *adaptor_pairwise([parameters['plane_n'], parameters['plane_c']]), 
            adaptor_P_gt_pairwise(P_gt)
        )

    def compute_residue_single(n, c, p):
        # n: ...x3, c: ..., p: ...x3
        return tf.square(tf.reduce_sum(p * n, axis=-1) - c)

    def compute_parameter_loss(parameters_pred, parameters_gt, matching_indices, angle_diff):
        # n - BxKx3
        n = batched_gather(parameters_pred['plane_n'], matching_indices, axis=1)
        dot_abs = tf.abs(tf.reduce_sum(n * parameters_gt['plane_n'], axis=2))
        if angle_diff:
            return acos_safe(dot_abs) # BxK
        else:
            return 1.0 - dot_abs # BxK

    def extract_predicted_parameters_as_json(fetched, k):
        # This is only for a single plane
        plane = Plane(fetched['plane_n'][k], fetched['plane_c'][k])
        
        json_info = {
            'type': 'plane',
            'center_x': plane.center[0],
            'center_y': plane.center[1],
            'center_z': plane.center[2], 
            'normal_x': plane.n[0],
            'normal_y': plane.n[1],
            'normal_z': plane.n[2],
            'x_size': plane.x_range[1] - plane.x_range[0],
            'y_size': plane.y_range[1] - plane.y_range[0],
            'x_axis_x': plane.x_axis[0],
            'x_axis_y': plane.x_axis[1],
            'x_axis_z': plane.x_axis[2],
            'y_axis_x': plane.y_axis[0],
            'y_axis_y': plane.y_axis[1],
            'y_axis_z': plane.y_axis[2],
        }

        return json_info

    def extract_parameter_data_as_dict(primitives, n_max_instances):
        n = np.zeros(dtype=float, shape=[n_max_instances, 3])
        for i, primitive in enumerate(primitives):
            if isinstance(primitive, Plane):
                n[i] = primitive.n
        return {
            'plane_n_gt': n
        }

    def create_primitive_from_dict(d):
        assert d['type'] == 'plane'
        location = np.array([d['location_x'], d['location_y'], d['location_z']], dtype=float)
        axis = np.array([d['axis_x'], d['axis_y'], d['axis_z']], dtype=float)
        return Plane(n=axis, c=np.dot(location, axis))

