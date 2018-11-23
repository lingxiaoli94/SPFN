import os, sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

import bundle_io
import fitter_factory

import numpy as np
import h5py
import pickle
import random
import json
import subprocess
import argparse

EXE = '/orionp2/deepprimitivefit/app/primitive-fitting-legacy/build/OSMesaRenderer'

def make_segmentation_cmd(point_cloud_path, point_labels_path, output_prefix):
    return [
        EXE, 
        '--disable_primitives_rendering', 
        '--point_cloud='+point_cloud_path,  
        '--point_labels='+point_labels_path, 
        '--snapshot='+output_prefix+'_segmentation', 
        '--azimuth_deg=-60', 
        '--elevation_deg=-20',
        '--theta_deg=0'
    ]

def make_mesh_rendering_cmd(point_cloud_path, point_labels_path, all_primitives_json_path, output_prefix):
    return [
        EXE, 
        '--run_primitive_meshing', 
        '--disable_point_cloud_rendering', 
        '--disable_primitives_rendering', 
        '--point_cloud='+point_cloud_path, 
        '--point_labels='+point_labels_path, 
        '--primitives='+all_primitives_json_path, 
        '--snapshot='+output_prefix, 
        '--azimuth_deg=-60', 
        '--elevation_deg=-20', 
        '--theta_deg=0', 
        '--out_mesh='+output_prefix+'.off', 
        '--out_face_labels='+output_prefix+'_out_face_labels.txt'
    ]

def make_primitive_rendering_cmd(point_cloud_path, point_values_path, primitives_json_path, output_prefix):
    return [
        EXE, 
        '--point_cloud='+point_cloud_path, 
        '--point_values='+point_values_path, 
        '--primitives='+primitives_json_path, 
        '--snapshot='+output_prefix, 
        '--azimuth_deg=-60', 
        '--elevation_deg=-20', 
        '--theta_deg=0'
    ]


def render_no_gt(pc_txt, pred_h5_file, render_each_primitive=False):
    # only call this for neural net predictions
    pred_prefix = os.path.splitext(pc_txt)[0]
    pred_h5 = h5py.File(pred_h5_file, 'r')
    W = pred_h5['instance_per_point'][()] # NxK
    name_to_id_dict = pickle.loads(pred_h5.attrs['name_to_id_dict'])
    primitive_id_to_ephemeral_id_dict = {name_to_id_dict[key]: fitter_factory.primitive_name_to_id(key) for key in name_to_id_dict}

    assert(len(W.shape) == 2)
    n_points = W.shape[0]
    n_max_instances = W.shape[1]

    instance_per_point = np.argmax(W, axis=1) # N
    type_per_point = np.array([primitive_id_to_ephemeral_id_dict[x] for x in pred_h5['type_per_point']]) # N

    primitive_json_dict = {}
    null_mask = np.zeros([n_points], dtype=bool)
    for k in range(n_max_instances):
        cur_mask = instance_per_point == k
        count = cur_mask.sum()
        if count < 0.005 * n_points:
            null_mask[k] = True
            continue

        member_type_list = np.array([type_per_point[i] for i in range(n_points) if cur_mask[i]])
        T_pred = np.bincount(member_type_list).argmax()
        
        fitter_cls = fitter_factory.all_fitter_classes[T_pred] # use predicted type
        json_data = fitter_cls.extract_predicted_parameters_as_json(pred_h5['parameters'], k)
        json_data['label'] = k + 1 # need to plus one here
        primitive_json_dict[k] = json_data

    point_cloud_path = pred_prefix + '_point_cloud.txt'
    subprocess.run(['cp', pc_txt, point_cloud_path])

    point_labels = np.array([(0 if null_mask[t] else t + 1) for t in instance_per_point])
    point_labels_path = pred_prefix + '_point_labels.txt'
    np.savetxt(point_labels_path, point_labels, delimiter=' ', fmt='%i')

    if render_each_primitive:
        # visualize each primitive
        for k in primitive_json_dict:
            point_values_path = pred_prefix + '_point_values_{}.txt'.format(k)
            primitives_json_path = pred_prefix + '_primitives_{}.json'.format(k)
            primitives_png_path = pred_prefix + '_primitives_{}'.format(k) # no suffix
            np.savetxt(point_values_path, W[:, k], delimiter=' ', fmt='%i')
            open(primitives_json_path, 'w').write(json.dumps([primitive_json_dict[k]], cls=bundle_io.NumpyEncoder))
            subprocess.run(make_primitive_rendering_cmd(point_cloud_path, point_values_path, primitives_json_path, primitives_png_path))

    all_primitives_json_path = pred_prefix + '_all_primitives.json'
    open(all_primitives_json_path, 'w').write(json.dumps(list(primitive_json_dict.values()), cls=bundle_io.NumpyEncoder))
    all_primitives_png_path_no_ext = pred_prefix + '_all_primitives'
    subprocess.run(make_mesh_rendering_cmd(point_cloud_path, point_labels_path, all_primitives_json_path, all_primitives_png_path_no_ext))
    subprocess.run(make_segmentation_cmd(point_cloud_path, point_labels_path, pred_prefix))

def render(bundle_prefix, render_each_primitive=False, use_pred_idx=False):
    bundle = bundle_io.load_single_bundle(bundle_prefix)
    fitter_factory.register_primitives(bundle['list_of_primitives'])

    matching_indices = bundle['matching_indices']
    loss_dict = bundle['loss_dict']
    stats = bundle['stats']
    data = bundle['data']
    pred_result = bundle['pred_result']
    null_mask = bundle['null_mask']
    mask_gt_nulled = bundle['mask_gt_nulled']
    instance_per_point = bundle['instance_per_point'] # N, can be -1 for unassigned points
    T_pred = bundle['type_per_instance'][()] # K, can be -1 for null primitives

    n_max_instances = T_pred.shape[0]
    n_points = instance_per_point.shape[0]

    pc_concat = np.concatenate([data['P'], data['normal_gt']], axis=1)
    point_cloud_path = bundle_prefix + '_point_cloud.txt'
    np.savetxt(point_cloud_path, pc_concat, delimiter=' ', fmt='%f %f %f %f %f %f')

    n_gt_instances = np.max(data['I_gt']) + 1
    loss_paths = []
    primitive_id_to_render_id = {}
    primitive_json_dict = {}
    for k in range(n_max_instances):
        if null_mask[k]: continue
        fitter_cls = fitter_factory.all_fitter_classes[T_pred[k]] # use predicted type
        json_data = fitter_cls.extract_predicted_parameters_as_json(pred_result['parameters'], k)
        k_mask = (matching_indices[:n_gt_instances] == k)
        if np.any(k_mask):
            # k is matched with some gt primitive
            json_data['label'] = int(np.argmax(k_mask)) + 1 if not use_pred_idx else k + 1 # need to plus one here
        else:
            # k is unmatched predicted primitives
            # assign random color
            json_data['label'] = int(random.randint(10000000, 99999999))
        primitive_id_to_render_id[k] = json_data['label']
        primitive_json_dict[k] = json_data

    point_labels = np.copy(instance_per_point)
    for i in range(point_labels.shape[0]):
        if point_labels[i] in primitive_id_to_render_id:
            point_labels[i] = primitive_id_to_render_id[point_labels[i]]
        else:
            point_labels[i] = 0
    point_labels_path = bundle_prefix + '_point_labels.txt'
    np.savetxt(point_labels_path, point_labels, delimiter=' ', fmt='%i')

    if render_each_primitive:
        W = np.eye(n_max_instances)[instance_per_point] # NxK, one_hot
        W[instance_per_point == -1] = np.zeros_like(W[0]) # zeroing out unassigned points
        # visualize each primitive
        for k in primitive_json_dict:
            point_values_path = bundle_prefix + '_point_values_{}.txt'.format(k)
            primitives_json_path = bundle_prefix + '_primitives_{}.json'.format(k)
            primitives_png_path = bundle_prefix + '_primitives_{}'.format(k) # no suffix
            np.savetxt(point_values_path, W[:, k], delimiter=' ', fmt='%i')
            open(primitives_json_path, 'w').write(json.dumps([primitive_json_dict[k]], cls=bundle_io.NumpyEncoder))
            subprocess.run(make_primitive_rendering_cmd(point_cloud_path, point_values_path, primitives_json_path, primitives_png_path))

    all_primitives_json_path = bundle_prefix + '_all_primitives.json'
    open(all_primitives_json_path, 'w').write(json.dumps(list(primitive_json_dict.values()), cls=bundle_io.NumpyEncoder))
    all_primitives_png_path_no_ext = bundle_prefix + '_all_primitives'
    # Finishing writing to files, start rendering
    rendering_mesh_cmd = make_mesh_rendering_cmd(point_cloud_path, point_labels_path, all_primitives_json_path, all_primitives_png_path_no_ext)
    rendering_seg_cmd = make_segmentation_cmd(point_cloud_path, point_labels_path, bundle_prefix)

    # order of creation is going to be reversed in sigal
    # subprocess.run([EXE, '--disable_primitives_rendering', '--point_cloud='+point_cloud_path,  '--snapshot='+bundle_prefix+'_pc', '--azimuth_deg=-60', '--elevation_deg=-20', '--theta_deg=0'])

    caption = ''
    for key in loss_dict.keys():
        if type(loss_dict[key]) == np.float32:
            caption += '{}: {}\n'.format(key, loss_dict[key])
    for key in stats.keys():
        if type(stats[key]) == np.float32:
            caption += '{}: {}\n'.format(key, stats[key])
    subprocess.run(['convert', '-background', 'Khaki', 'label:{}'.format(caption), '-gravity', 'Center', '{}_stats.png'.format(bundle_prefix)])

    subprocess.run(rendering_seg_cmd)
    subprocess.run(rendering_mesh_cmd)

    cmds = ' '.join(rendering_seg_cmd) + '\n' + ' '.join(rendering_mesh_cmd)
    open(bundle_prefix + '_cmd.txt', 'w').write(cmds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pc_txt', type=str)
    parser.add_argument('pred_h5_file', type=str)
    parser.add_argument('--render_each_primitive', action='store_true')
    args = parser.parse_args()

    fitter_factory.register_primitives(['plane', 'sphere', 'cylinder', 'cone'])
    render_no_gt(args.pc_txt, args.pred_h5_file, args.render_each_primitive)
