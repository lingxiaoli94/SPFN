import os, sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
import fitter_factory

import numpy as np
import pandas
import h5py
import pickle

class PredictionLoader:
    def __init__(self, n_max_instances, pred_dir):
        self.n_max_instances = n_max_instances
        self.basename_to_hdf5_file = {}
        hdf5_file_list = os.listdir(pred_dir)
        basename_list = [os.path.splitext(os.path.basename(p))[0] for p in hdf5_file_list]
        for idx, h5_file in enumerate(hdf5_file_list):
            basename = os.path.splitext(os.path.basename(h5_file))[0]
            self.basename_to_hdf5_file[basename] = h5_file

    def load_multiple(self, basename_list):
        batch_size = len(basename_list)
        method_name_list = []
        result = {
            'normal_per_point': [],
            'instance_per_point': [],
            'type_per_point': [],
            'parameters': {},
        }

        for idx, basename in enumerate(basename_list):
            hdf5_file = self.basename_to_hdf5_file[basename]
            f = h5py.File(hdf5_file, 'r')
            result['normal_per_point'].append(f['normal_per_point'][()])
            result['instance_per_point'].append(f['instance_per_point'][()])
            method_name_list.append(str(f.attrs['method_name']))
            primitive_name_to_id_dict = pickle.loads(f.attrs['name_to_id_dict'])
            primitive_id_to_ephemeral_id_dict = {primitive_name_to_id_dict[key]: fitter_factory.primitive_name_to_id(key) for key in primitive_name_to_id_dict}
            result['type_per_point'].append(np.array([primitive_id_to_ephemeral_id_dict[x] for x in f['type_per_point']]))

            for key in f['parameters']:
                if key not in result['parameters']:
                    result['parameters'][key] = []
                params = f['parameters'][key][()]
                # add paddings
                if params.shape[0] < self.n_max_instances:
                    diff = self.n_max_instances - params.shape[0]
                    if len(params.shape) == 1:
                        params = np.pad(params, pad_width=((0, diff)), mode='constant', constant_values=0)
                    elif len(params.shape) == 2:
                        params = np.pad(params, pad_width=((0, diff), (0, 0)), mode='constant', constant_values=0)
                    else:
                        assert False
                result['parameters'][key].append(params)
        
        packed = {}
        for key in result.keys():
            if key != 'parameters':
                packed[key] = np.stack(result[key], axis=0)
            else:
                packed['parameters'] = {}
                for key2 in result['parameters'].keys():
                    packed['parameters'][key2] = np.stack(result['parameters'][key2], axis=0)
        return packed, method_name_list
        
def save_batch_nn(nn_name, pred_result, basename_list, save_dir, W_reduced=True):
    batch_size = pred_result['W'].shape[0]
    assert batch_size == len(basename_list)
    type_per_point = np.argmax(pred_result['type_per_point'], axis=2) # BxN
    instance_per_point = pred_result['W'] # BxNxK
    if W_reduced:
        instance_per_point = np.argmax(instance_per_point, axis=2) # BxN
    for b in range(batch_size):
        f = h5py.File(os.path.join(save_dir, basename_list[b] + '.h5'), 'w')
        f.attrs['method_name'] = nn_name
        f.attrs['basename'] = basename_list[b]
        f.attrs['name_to_id_dict'] = np.void(pickle.dumps(fitter_factory.primitive_name_to_id_dict))
        f.create_dataset('normal_per_point', data=pred_result['normal_per_point'][b])
        f.create_dataset('type_per_point', data=type_per_point[b])
        f.create_dataset('instance_per_point', data=instance_per_point[b])
        g = f.create_group('parameters')
        for key in pred_result['parameters']:
            g.create_dataset(key, data=pred_result['parameters'][key][b])

def save_single_nn(nn_name, pred_result, pred_h5_file, W_reduced=True):
    batch_size = pred_result['W'].shape[0]
    assert batch_size == 1
    type_per_point = np.argmax(pred_result['type_per_point'], axis=2) # BxN
    instance_per_point = pred_result['W'] # BxNxK
    if W_reduced:
        instance_per_point = np.argmax(instance_per_point, axis=2) # BxN
    f = h5py.File(pred_h5_file, 'w')
    f.attrs['method_name'] = nn_name
    f.attrs['name_to_id_dict'] = np.void(pickle.dumps(fitter_factory.primitive_name_to_id_dict))
    f.create_dataset('normal_per_point', data=pred_result['normal_per_point'][0])
    f.create_dataset('type_per_point', data=type_per_point[0])
    f.create_dataset('instance_per_point', data=instance_per_point[0])
    g = f.create_group('parameters')
    for key in pred_result['parameters']:
        g.create_dataset(key, data=pred_result['parameters'][key][0])
