import os, sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '..', '..'))
sys.path.append(os.path.join(BASE_DIR, '..', '..', '..'))

from utils.tf_wrapper import batched_gather
from utils.geometry_utils import weighted_sphere_fitting
from utils.tf_numerical_safe import sqrt_safe
from fitters.adaptors import *
from primitives import Sphere

import tensorflow as tf
import numpy as np

class SphereFitter:
    def primitive_name():
        return 'sphere'

    def insert_prediction_placeholders(pred_ph, n_max_instances):
        pred_ph['sphere_center'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances, 3])
        pred_ph['sphere_radius_squared'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances])

    def normalize_parameters(parameters):
        parameters['sphere_radius_squared'] = tf.clip_by_value(parameters['sphere_radius_squared'], 1e-4, 1e6)

    def insert_gt_placeholders(parameters_gt, n_max_instances):
        pass

    def fill_gt_placeholders(feed_dict, parameters_gt, batch):
        pass

    def compute_parameters(feed_dict, parameters):
        P = feed_dict['P']
        W = feed_dict['W']
        batch_size = tf.shape(P)[0]
        n_points = tf.shape(P)[1]
        n_max_primitives = tf.shape(W)[2]
        P = tf.tile(tf.expand_dims(P, axis=1), [1, n_max_primitives, 1, 1]) # BxKxNx3
        W = tf.transpose(W, perm=[0, 2, 1]) # BxKxN
        P = tf.reshape(P, [batch_size * n_max_primitives, n_points, 3]) # BKxNx3
        W = tf.reshape(W, [batch_size * n_max_primitives, n_points]) # BKxN
        fitting_result = weighted_sphere_fitting(P, W)

        parameters['sphere_center'] = tf.reshape(fitting_result['center'], [batch_size, n_max_primitives, 3])
        parameters['sphere_radius_squared'] = tf.reshape(fitting_result['radius_squared'], [batch_size, n_max_primitives])

    def compute_residue_loss(parameters, P_gt, matching_indices):
        return SphereFitter.compute_residue_single(
            *adaptor_matching([parameters['sphere_center'], parameters['sphere_radius_squared']], matching_indices), 
            P_gt
        )

    def compute_residue_loss_pairwise(parameters, P_gt):
        return SphereFitter.compute_residue_single(
            *adaptor_pairwise([parameters['sphere_center'], parameters['sphere_radius_squared']]), 
            adaptor_P_gt_pairwise(P_gt)
        )

    def compute_residue_single(center, radius_squared, p):
        return tf.square(sqrt_safe(tf.reduce_sum(tf.square(p - center), axis=-1)) - sqrt_safe(radius_squared))

    def compute_parameter_loss(parameters_pred, parameters_gt, matching_indices, angle_diff):
        return None

    def extract_parameter_data_as_dict(primitives, n_max_primitives):
        return {}

    def extract_predicted_parameters_as_json(fetched, k):
        sphere = Sphere(fetched['sphere_center'][k], np.sqrt(fetched['sphere_radius_squared'][k]))

        return {
            'type': 'sphere',
            'center_x': sphere.center[0],
            'center_y': sphere.center[1],
            'center_z': sphere.center[2],
            'radius': sphere.radius,
        }

    def create_primitive_from_dict(d):
        assert d['type'] == 'sphere'
        location = np.array([d['location_x'], d['location_y'], d['location_z']], dtype=float)
        radius = float(d['radius'])
        return Sphere(center=location, radius=radius)
