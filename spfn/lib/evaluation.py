import os, sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))
import fitter_factory
from utils.tf_wrapper import batched_gather
from utils.tf_numerical_safe import acos_safe, sqrt_safe
from constants import DIVISION_EPS

import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment

def create_gt_dict(n_max_instances):
    '''
        Returns gt_dict containing:
            - instance_per_point: BxN
            - normal_per_point: BxNx3
            - type_per_instance: BxK
            - points_per_instance: BxKxN'x3, sampled points on each instance
            - parameters: a dict, each entry is a BxKx... tensor
    '''
    # create gt_dict
    gt_dict = {}
    gt_dict['instance_per_point'] = tf.placeholder(dtype=tf.int32, shape=[None, None])
    gt_dict['normal_per_point'] = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
    gt_dict['type_per_instance'] = tf.placeholder(dtype=tf.int32, shape=[None, n_max_instances])
    gt_dict['points_per_instance'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances, None, 3])
    gt_dict['parameters'] = {}
    for fitter_cls in fitter_factory.get_all_fitter_classes():
        fitter_cls.insert_gt_placeholders(gt_dict['parameters'], n_max_instances=n_max_instances)

    return gt_dict

def fill_gt_dict_with_batch_data(feed_dict, gt_dict, batch):
    feed_dict.update({
        gt_dict['points_per_instance']: batch['P_gt'], 
        gt_dict['normal_per_point']: batch['normal_gt'],
        gt_dict['instance_per_point']: batch['I_gt'], 
        gt_dict['type_per_instance']: batch['T_gt'], 
    })
    for fitter_cls in fitter_factory.get_all_fitter_classes():
        fitter_cls.fill_gt_placeholders(feed_dict, gt_dict['parameters'], batch)

def nn_filter_W(W, W_null_threshold=0.005):
    # for SPFN & DPPN output, we do two postprocessings: 1) nullify columns with too few points 2) make entries of W binary

    n_points = tf.shape(W)[1]
    n_max_instances = W.get_shape()[2] # n_max_instances should not be dynamic

    W_column_sum = tf.reduce_sum(W, axis=1) # BxK
    null_mask = W_column_sum < tf.to_float(n_points) * W_null_threshold # BxK
    null_mask_W_like = tf.tile(tf.expand_dims(null_mask, axis=1), [1, n_points, 1]) # BxNxK
    W = tf.where(null_mask_W_like, tf.zeros_like(W), tf.one_hot(tf.argmax(W, axis=2), depth=n_max_instances, dtype=tf.float32))

    return W

def evaluate(pred_dict, gt_dict, is_eval, is_nn, P_in=None):
    '''
        Input: 
            pred_dict should contain:
                - W: BxNxK, segmentation instances. Allow zero rows to indicate unassigned points.
                - normal_per_point: BxNx3, normal per point
                - type_per_point: type per points
                    - This should be logit of shape BxNxT if is_eval=False, and actual value of shape BxN otherwise
                    - can contain -1
                - parameters - a dict, each entry is a BxKx... tensor
            gt_dict should be obtained from calling create_gt_dict
            P_in - BxNx3 is the input point cloud, used only when is_eval=True

        Returns: {loss_dict, matching_indices} + stats from calculate_eval_stats(), where
            - loss_dict contains:
                - normal_loss: B, averaged over all N points
                - type_loss: B, averaged over all N points. 
                    - This is cross entropy loss during training, and accuracy during test time
                - miou_loss: BxK, mean IoU loss for each matched instances
                - residue_loss: BxK, residue loss for each instance
                - parameter_loss: BxK, parameter loss for each instance
                - avg_miou_loss: B
                - avg_residue_loss: B
                - avg_parameter_loss: B
            - matching_indices: BxK, where (b,k)th ground truth primitive is matched with (b, matching_indices[b, k])
    '''
    # dimension tensors
    W = pred_dict['W']
    batch_size = tf.shape(W)[0] 
    n_points = tf.shape(W)[1]
    n_max_instances = W.get_shape()[2] # n_max_instances should not be dynamic
    n_registered_primitives = fitter_factory.get_n_registered_primitives()

    if is_eval and is_nn:
        # at evaluation, want W to be binary and filtered (if is from nn)
        W = nn_filter_W(W)

    # shortcuts
    # note that I_gt can contain -1, indicating instance of unknown primitive type
    I_gt = gt_dict['instance_per_point'] # BxN
    T_gt = gt_dict['type_per_instance'] # BxK

    n_instances_gt = tf.reduce_max(I_gt, axis=1) + 1 # only count known primitive type instances, as -1 will be ignored
    mask_gt = tf.sequence_mask(n_instances_gt, maxlen=n_max_instances) # BxK, mask_gt[b, k] = 1 iff instace k is present in the ground truth batch b

    matching_indices = tf.stop_gradient(tf.py_func(hungarian_matching, [W, I_gt], Tout=tf.int32)) # BxK
    miou_loss = compute_miou_loss(W, I_gt, matching_indices) # losses all have dimension BxK
    normal_loss = compute_normal_loss(pred_dict['normal_per_point'], gt_dict['normal_per_point'], angle_diff=is_eval) # B
    per_point_type_loss = compute_per_point_type_loss(pred_dict['type_per_point'], I_gt, T_gt, is_eval=is_eval) # B

    residue_losses = [] # a length T array of BxK tensors
    parameter_losses = [] # a length T array of BxK tensors
    residue_per_point_array = [] # a length T array of BxKxN' tensors
    for fitter_cls in fitter_factory.get_all_fitter_classes():
        residue_per_point = fitter_cls.compute_residue_loss(pred_dict['parameters'], gt_dict['points_per_instance'], matching_indices) # BxKxN'
        residue_per_point_array.append(residue_per_point)
        residue_losses.append(tf.reduce_mean(residue_per_point, axis=2))
        parameter_loss = fitter_cls.compute_parameter_loss(pred_dict['parameters'], gt_dict['parameters'], matching_indices, angle_diff=is_eval)
        if parameter_loss is None:
            parameter_loss = tf.zeros(dtype=tf.float32, shape=[batch_size, n_max_instances])
        parameter_losses.append(parameter_loss)
    residue_losses = tf.stack(residue_losses, axis=2)
    parameter_losses = tf.stack(parameter_losses, axis=2)
    residue_per_point_array = tf.stack(residue_per_point_array, axis=3) # BxKxN'xT

    # Aggregate losses across fitters
    residue_loss = aggregate_loss_from_stacked(residue_losses, T_gt) # BxK
    parameter_loss = aggregate_loss_from_stacked(parameter_losses, T_gt) # BxK

    loss_dict = {
        'normal_loss': normal_loss,
        'type_loss': per_point_type_loss,
        'miou_loss': miou_loss,
        'residue_loss': residue_loss,
        'parameter_loss': parameter_loss,
        'avg_miou_loss': reduce_mean_masked_instance(miou_loss, mask_gt),
        'avg_residue_loss': reduce_mean_masked_instance(residue_loss, mask_gt),
        'avg_parameter_loss': reduce_mean_masked_instance(parameter_loss, mask_gt),
    }

    result = {'loss_dict': loss_dict, 'matching_indices': matching_indices}

    if is_eval:
        result.update(
            calculate_eval_stats( 
                W=W,
                matching_indices=matching_indices,
                mask_gt=mask_gt,
                P_in=P_in,
                type_per_point=pred_dict['type_per_point'],
                T_gt=T_gt,
                parameters=pred_dict['parameters'],
                residue_losses=residue_losses,
                parameter_loss=parameter_loss,
                residue_per_point_array=residue_per_point_array,
            )
        )

    return result

def calculate_eval_stats(W, matching_indices, mask_gt, P_in, type_per_point, T_gt, parameters, residue_losses, parameter_loss, residue_per_point_array):
    '''
        Returns a dict containing:
            - stats : {
                per_instance_type_accuracy: B, average primitive type accuracy for a shape
                avg_residue_loss_without_gt: B, average residue loss using the predicted type
                parameter_loss_without_gt: B, average parameter loss using the predicted type (over only primitives with matched type)
            }
            - null_mask: BxK, indicating which predicted primitives are null
            - mask_gt_nulled: BxK, indicated which ground truth primitive is not null and is matched with a predicted (non-null) primitive
            - instance_per_point: BxN, non-one-hot version of W
            - type_per_intance: BxK, type for predicted primitives
            - residue_gt_primitive: BxKxN', distance from sampled points on ground truth S_k to the predicted primitive matched with S_k
            - residue_to_closest: BxN, distance from each input point to the closest predicted primitive
    '''
    batch_size = tf.shape(W)[0] 
    n_points = tf.shape(W)[1]
    n_max_instances = W.get_shape()[2] # n_max_instances should not be dynamic
    n_registered_primitives = fitter_factory.get_n_registered_primitives()

    null_mask = tf.reduce_sum(W, axis=1) < 0.5 # BxK
    # null_mask indicates which predicted primitives are null
    I = tf.where(tf.reduce_sum(W, axis=2) > 0.5, tf.argmax(W, axis=2, output_type=tf.int32), tf.fill([batch_size, n_points], -1)) # BxN
    # I can have -1 entries, indicating unassigned points, just like I_gt, and tf.one_hot(I) == W

    per_point_type_one_hot = tf.one_hot(type_per_point, depth=n_registered_primitives, dtype=tf.float32) # BxNxT
    instance_type_prob = tf.reduce_sum(tf.expand_dims(W, axis=3) * tf.expand_dims(per_point_type_one_hot, axis=2), axis=1) # BxKxT
    instance_type = tf.argmax(instance_type_prob, axis=2, output_type=tf.int32) # BxK


    null_mask_gt = batched_gather(null_mask, matching_indices, axis=1) # BxK, indicating which gt primitive is not matched
    mask_gt_nulled = tf.logical_and(mask_gt, tf.logical_not(null_mask_gt)) # only count these gt primitives towards some metrics
    residue_loss_without_gt = aggregate_loss_from_stacked(residue_losses, batched_gather(instance_type, matching_indices, axis=1)) # BxK
    avg_residue_loss_without_gt = reduce_mean_masked_instance(residue_loss_without_gt, mask_gt_nulled) # B

    # for parameter loss w/o gt, only count when the predicted type matches the ground truth type
    instance_matched_mask = tf.equal(T_gt, batched_gather(instance_type, matching_indices, axis=1)) # BxK, boolean
    per_instance_type_accuracy = reduce_mean_masked_instance(tf.to_float(instance_matched_mask), mask_gt_nulled) # B
    parameter_loss_without_gt = reduce_mean_masked_instance(parameter_loss, tf.logical_and(instance_matched_mask, mask_gt_nulled))

    result = {
        'stats': {
            'per_instance_type_accuracy': per_instance_type_accuracy, # B
            'avg_residue_loss_without_gt': avg_residue_loss_without_gt, # B
            'parameter_loss_without_gt': parameter_loss_without_gt, # B
        },
    }

    residue_matrix = []
    identity_matching_indices = tf.tile(tf.expand_dims(tf.range(n_max_instances), axis=0), [batch_size, 1]) # BxK
    for fitter_cls in fitter_factory.get_all_fitter_classes():
        residue_per_point = fitter_cls.compute_residue_loss(parameters, tf.tile(tf.expand_dims(P_in, axis=1), [1, n_max_instances, 1, 1]), matching_indices=identity_matching_indices) # BxKxN
        residue_matrix.append(residue_per_point)
    residue_matrix = tf.stack(residue_matrix, axis=3) # BxKxNxT, this matrix might be large!

    # residue_to_primitive[b,n,k] = residue_matrix[b, k, n, instance_type[b,k]] 
    indices_0 = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(batch_size), axis=1), axis=2), [1, n_points, n_max_instances]) # BxNxK
    indices_1 = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(n_max_instances), axis=0), axis=1), [batch_size, n_points, 1]) # BxNxK
    indices_2 = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(n_points), axis=0), axis=2), [batch_size, 1, n_max_instances]) # BxNxK
    indices_3 = tf.tile(tf.expand_dims(instance_type, axis=1), [1, n_points, 1]) # BxNxK
    residue_to_primitive = tf.gather_nd(residue_matrix, indices=tf.stack([indices_0, indices_1, indices_2, indices_3], axis=3)) # BxNxK
    # set null primitive residues to a large number
    null_mask_W_like = tf.tile(tf.expand_dims(null_mask, axis=1), [1, n_points, 1]) # BxNxK
    residue_to_primitive = tf.where(null_mask_W_like, 1e8*tf.ones_like(residue_to_primitive), residue_to_primitive) # BxNxK
    residue_to_closest = tf.reduce_min(residue_to_primitive, axis=2) # BxN

    residue_gt_primitive = aggregate_per_point_loss_from_stacked(residue_per_point_array, batched_gather(instance_type, matching_indices, axis=1)) # BxKxN', squared distance

    # Save information for downstream analysis
    result['null_mask'] = null_mask # BxK
    result['mask_gt_nulled'] = mask_gt_nulled # BxK
    result['instance_per_point'] = I # BxN
    result['type_per_instance'] = instance_type # BxK
    result['residue_gt_primitive'] = tf.sqrt(residue_gt_primitive) # BxKxN'
    result['residue_to_closest'] = tf.sqrt(residue_to_closest) # BxN

    return result

def aggregate_loss_from_stacked(loss_stacked, T_gt):
    # loss_stacked - BxKxT, T_gt - BxK
    # out[b, k] = loss_stacked[b, k, T_gt[b, k]]
    B = tf.shape(loss_stacked)[0]
    K = tf.shape(loss_stacked)[1]
    indices_0 = tf.tile(tf.expand_dims(tf.range(B), axis=1), multiples=[1, K]) # BxK
    indices_1 = tf.tile(tf.expand_dims(tf.range(K), axis=0), multiples=[B, 1]) # BxK
    indices = tf.stack([indices_0, indices_1, T_gt], axis=2) # BxKx3
    return tf.gather_nd(loss_stacked, indices=indices)

def aggregate_per_point_loss_from_stacked(loss_stacked, T_gt):
    # loss_stacked - BxKxN'xT, T_gt - BxK
    # out[b, k, n'] = loss_stacked[b, k, n', T_gt[b, k]]
    B = tf.shape(loss_stacked)[0]
    K = tf.shape(loss_stacked)[1]
    N_p = tf.shape(loss_stacked)[2]

    indices_0 = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(B), axis=1), axis=2), multiples=[1, K, N_p]) # BxKxN'
    indices_1 = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(K), axis=0), axis=2), multiples=[B, 1, N_p]) # BxKxN'
    indices_2 = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(N_p), axis=0), axis=0), multiples=[B, K, 1]) # BxKxN'
    indices_3 = tf.tile(tf.expand_dims(T_gt, axis=2), multiples=[1, 1, N_p]) # BxKxN'
    indices = tf.stack([indices_0, indices_1, indices_2, indices_3], axis=3)
    return tf.gather_nd(loss_stacked, indices=indices) # BxKxN'

def reduce_mean_masked_instance(loss, mask_gt):
    # loss: BxK
    loss = tf.where(mask_gt, loss, tf.zeros_like(loss))
    reduced_loss = tf.reduce_sum(loss, axis=1) # B
    denom = tf.reduce_sum(tf.to_float(mask_gt), axis=1) # B
    return tf.where(denom > 0, reduced_loss / denom, tf.zeros_like(reduced_loss)) # B

def compute_normal_loss(normal, normal_gt, angle_diff):
    # normal, normal_gt: BxNx3
    # Assume normals are unoriented
    dot_abs = tf.abs(tf.reduce_sum(normal * normal_gt, axis=2)) # BxN
    if angle_diff:
        return tf.reduce_mean(acos_safe(dot_abs), axis=1)
    else:
        return tf.reduce_mean(1.0 - dot_abs, axis=1)

def compute_miou_loss(W, I_gt, matching_indices):
    # W - BxNxK
    # I_gt - BxN
    W_reordered = batched_gather(W, indices=matching_indices, axis=2) # BxNxK
    depth = tf.shape(W)[2]
    # notice in tf.one_hot, -1 will result in a zero row, which is what we want
    W_gt = tf.one_hot(I_gt, depth=depth, dtype=tf.float32) # BxNxK
    dot = tf.reduce_sum(W_gt * W_reordered, axis=1) # BxK
    denominator = tf.reduce_sum(W_gt, axis=1) + tf.reduce_sum(W_reordered, axis=1) - dot
    mIoU = dot / (denominator + DIVISION_EPS) # BxK
    return 1.0 - mIoU

def compute_per_point_type_loss(per_point_type, I_gt, T_gt, is_eval):
    # For training, per_point_type is BxNxQ, where Q = n_registered_primitives
    # For test, per_point_type is BxN
    # I_gt - BxN, allow -1
    # T_gt - BxK
    batch_size = tf.shape(I_gt)[0]
    n_points = tf.shape(I_gt)[1]
    indices_0 = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, n_points]) # BxN
    indices = tf.stack([indices_0, tf.maximum(0, I_gt)], axis=2)
    per_point_type_gt = tf.gather_nd(T_gt, indices=indices) # BxN
    if is_eval:
        type_loss = 1.0 - tf.to_float(tf.equal(per_point_type, per_point_type_gt))
    else:
        type_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=per_point_type, labels=per_point_type_gt) # BxN

    # do not add loss to background points in gt
    type_loss = tf.where(tf.equal(I_gt, -1), tf.zeros_like(type_loss), type_loss)
    return tf.reduce_sum(type_loss, axis=1) / tf.to_float(tf.count_nonzero(tf.not_equal(I_gt, -1), axis=1)) # B

def hungarian_matching(W_pred, I_gt):
    # This non-tf function does not backprob gradient, only output matching indices
    # W_pred - BxNxK
    # I_gt - BxN, may contain -1's
    # Output: matching_indices - BxK, where (b,k)th ground truth primitive is matched with (b, matching_indices[b, k])
    #   where only n_gt_labels entries on each row have meaning. The matching does not include gt background instance
    batch_size = I_gt.shape[0]
    n_points = I_gt.shape[1]
    n_max_labels = W_pred.shape[2]

    matching_indices = np.zeros([batch_size, n_max_labels], dtype=np.int32)
    for b in range(batch_size):
        # assuming I_gt[b] does not have gap
        n_gt_labels = np.max(I_gt[b]) + 1 # this is K'
        W_gt = np.zeros([n_points, n_gt_labels + 1]) # HACK: add an extra column to contain -1's
        W_gt[np.arange(n_points), I_gt[b]] = 1.0 # NxK'
        
        dot = np.sum(np.expand_dims(W_gt, axis=2) * np.expand_dims(W_pred[b], axis=1), axis=0) # K'xK
        denominator = np.expand_dims(np.sum(W_gt, axis=0), axis=1) + np.expand_dims(np.sum(W_pred[b], axis=0), axis=0) - dot
        cost = dot / np.maximum(denominator, DIVISION_EPS) # K'xK
        cost = cost[:n_gt_labels, :] # remove last row, corresponding to matching gt background instance

        _, col_ind = linear_sum_assignment(-cost) # want max solution
        matching_indices[b, :n_gt_labels] = col_ind

    return matching_indices

