import os, sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
from primitives import Plane

import numpy as np
import subprocess
import json
import h5py
import pickle

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32): 
            return np.asscalar(obj)
        return json.JSONEncoder.default(self, obj)

def dump_single_bundle(data, pred_result, loss_dict, stats, matching_indices, null_mask, mask_gt_nulled, instance_per_point, type_per_instance, residue_to_closest, residue_gt_primitive, list_of_primitives, method_name, output_prefix):
    f = h5py.File(output_prefix + '_bundle.h5', 'w')
    print('Dumping data to {}_bundle.h5'.format(output_prefix))
    data_gp = f.create_group('data')
    for key in data.keys():
        data_gp.create_dataset(key, data=data[key], compression='gzip')
    pred_gp = f.create_group('pred_result')
    pred_gp.attrs['method_name'] = method_name
    for key in pred_result.keys():
        if key != 'parameters':
            current_data = pred_result[key]
            pred_gp.create_dataset(key, data=current_data, compression='gzip')
        else:
            param_gp = pred_gp.create_group('parameters')
            for key2 in pred_result['parameters']:
                param_gp.create_dataset(key2, data=pred_result['parameters'][key2], compression='gzip')
    f.attrs['loss_dict'] = np.void(pickle.dumps(loss_dict))
    f.attrs['stats'] = np.void(pickle.dumps(stats))
    f.create_dataset('matching_indices', data=matching_indices, compression='gzip')
    f.create_dataset('null_mask', data=null_mask, compression='gzip')
    f.create_dataset('mask_gt_nulled', data=mask_gt_nulled, compression='gzip')
    f.create_dataset('instance_per_point', data=instance_per_point, compression='gzip')
    f.create_dataset('type_per_instance', data=type_per_instance, compression='gzip')
    f.create_dataset('residue_to_closest', data=residue_to_closest, compression='gzip')
    f.create_dataset('residue_gt_primitive', data=residue_gt_primitive, compression='gzip')
    f.attrs['list_of_primitives'] = np.void(pickle.dumps(list_of_primitives))

def load_single_bundle(input_prefix):
    filename = input_prefix + '_bundle.h5'
    print('Loading ' + filename + '...')
    f = h5py.File(filename, 'r')
    all_result = {}
    all_result['list_of_primitives'] = pickle.loads(f.attrs['list_of_primitives'])
    all_result['matching_indices'] = f['matching_indices'][()]
    all_result['mask_gt_nulled'] = f['mask_gt_nulled'][()]
    all_result['instance_per_point'] = f['instance_per_point'][()]
    all_result['type_per_instance'] = f['type_per_instance'][()]
    all_result['null_mask'] = f['null_mask'][()]
    all_result['residue_to_closest'] = f['residue_to_closest'][()]
    all_result['residue_gt_primitive'] = f['residue_gt_primitive'][()]
    all_result['loss_dict'] = pickle.loads(f.attrs['loss_dict'])
    all_result['stats'] = pickle.loads(f.attrs['stats'])
    all_result['data'] = {}
    for key in list(f['data'].keys()):
        all_result['data'][key] = f['data'][key]
    all_result['pred_result'] = {'method_name': str(f['pred_result'].attrs['method_name'])}
    for key in list(f['pred_result'].keys()):
        if key != 'parameters':
            all_result['pred_result'][key] = f['pred_result'][key]
        else:
            all_result['pred_result']['parameters'] = {}
            for key2 in list(f['pred_result']['parameters']):
                all_result['pred_result']['parameters'][key2] = f['pred_result']['parameters'][key2]
    return all_result
