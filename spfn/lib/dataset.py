import fitter_factory

import numpy as np
import random
import os
import h5py
import pickle
import pandas
import re

def create_unit_data_from_hdf5(f, n_max_instances, noisy, fixed_order=False, check_only=False, shuffle=True):
    ''' 
        f will be a h5py group-like object
    '''

    P = f['noisy_points'][()] if noisy else f['gt_points'][()] # Nx3
    normal_gt = f['gt_normals'][()] # Nx3
    I_gt = f['gt_labels'][()] # N

    P_gt = []

    # next check if soup_ids are consecutive
    found_soup_ids = []
    soup_id_to_key = {}
    soup_prog = re.compile('(.*)_soup_([0-9]+)$')
    for key in list(f.keys()):
        m = soup_prog.match(key)
        if m is not None:
            soup_id = int(m.group(2))
            found_soup_ids.append(soup_id)
            soup_id_to_key[soup_id] = key
    found_soup_ids.sort()
    n_instances = len(found_soup_ids)
    if n_instances == 0:
        return None
    for i in range(n_instances):
        if i not in found_soup_ids:
            print('{} is not found in soup ids!'.format(i))
            return None

    instances = []
    for i in range(n_instances):
        g = f[soup_id_to_key[i]]
        P_gt_cur = g['gt_points'][()]
        P_gt.append(P_gt_cur)
        meta = pickle.loads(g.attrs['meta'])
        primitive = fitter_factory.create_primitive_from_dict(meta)
        if primitive is None:
            return None
        instances.append(primitive)

    if n_instances > n_max_instances:
        print('n_instances {} > n_max_instances {}'.format(n_instances, n_max_instances))
        return None

    if np.amax(I_gt) >= n_instances:
        print('max label {} > n_instances {}'.format(np.amax(I_gt), n_instances))
        return None

    if check_only:
        return True

    T_gt = [fitter_factory.primitive_name_to_id(primitive.get_primitive_name()) for primitive in instances]
    T_gt.extend([0 for _ in range(n_max_instances - n_instances)]) # K

    n_total_points = P.shape[0]
    n_gt_points_per_instance = P_gt[0].shape[0]
    P_gt.extend([np.zeros(dtype=float, shape=[n_gt_points_per_instance, 3]) for _ in range(n_max_instances - n_instances)])

    # convert everything to numpy array
    P_gt = np.array(P_gt)
    T_gt = np.array(T_gt)
    
    if shuffle:
        # shuffle per point information around
        perm = np.random.permutation(n_total_points)
        P = P[perm]
        normal_gt = normal_gt[perm]
        I_gt = I_gt[perm]

    result = {
        'P': P,
        'normal_gt': normal_gt,
        'P_gt': P_gt,
        'I_gt': I_gt,
        'T_gt': T_gt,
    }

    # Next put in primitive parameters
    for fitter_cls in fitter_factory.all_fitter_classes:
        result.update(fitter_cls.extract_parameter_data_as_dict(instances, n_max_instances))

    return result

class Dataset:
    def __init__(self, batch_size, n_max_instances, csv_path, noisy, first_n=-1, fixed_order=False):
        self.batch_size = batch_size
        self.n_max_instances = n_max_instances
        self.fixed_order = fixed_order
        self.first_n = first_n
        self.noisy = noisy

        self.csv_raw = pandas.read_csv(csv_path, delimiter=',', header=None)
        self.hdf5_file_list = list(self.csv_raw[0])

        # make relative path absolute
        csv_folder = os.path.dirname(csv_path)
        self.hdf5_file_list = [os.path.join(csv_folder, p) for p in self.hdf5_file_list if not os.path.isabs(p)]
        
        if not fixed_order:
            random.shuffle(self.hdf5_file_list)
        if first_n != -1:
            self.hdf5_file_list = self.hdf5_file_list[:first_n]

        self.basename_list = [os.path.splitext(os.path.basename(p))[0] for p in self.hdf5_file_list]
        self.n_data = len(self.hdf5_file_list)
        self.first_iteration_finished = False

    def fetch_data_at_index(self, i):
        assert not self.first_iteration_finished
        path = self.hdf5_file_list[i]
        name = os.path.splitext(os.path.basename(path))[0]
        with h5py.File(path, 'r') as handle:
            data = create_unit_data_from_hdf5(handle, self.n_max_instances, noisy=self.noisy, fixed_order=self.fixed_order, shuffle=not self.fixed_order)
            assert data is not None # assume data are all clean

        return data

    def __iter__(self):
        self.current = 0
        if not self.fixed_order and self.first_iteration_finished:
            # shuffle data matrix
            perm = np.random.permutation(self.n_data)
            for key in self.data_matrix.keys():
                self.data_matrix[key] = self.data_matrix[key][perm]

        return self

    def __next__(self):
        if self.current >= self.n_data:
            if not self.first_iteration_finished:
                self.first_iteration_finished = True
            raise StopIteration()

        step = min(self.n_data - self.current, self.batch_size)
        assert step > 0
        self.last_step_size = step
        batched_data = {}
        if self.first_iteration_finished:
            for key in self.data_matrix.keys():
                batched_data[key] = self.data_matrix[key][self.current:self.current+step, ...]
        else:
            data = []
            for i in range(step):
                data.append(self.fetch_data_at_index(self.current + i))

            if not hasattr(self, 'data_matrix'):
                self.data_matrix = {}
                for key in data[0].keys():
                    trailing_ones = np.full([len(data[0][key].shape)], 1, dtype=int)
                    self.data_matrix[key] = np.tile(np.expand_dims(np.zeros_like(data[0][key]), axis=0), [self.n_data, *trailing_ones])
            for key in data[0].keys():
                batched_data[key] = np.stack([x[key] for x in data], axis=0)
                self.data_matrix[key][self.current:self.current+step, ...] = batched_data[key]

        self.current += step
        return batched_data

    def get_last_batch_range(self):
        # return: [l, r)
        return (self.current - self.last_step_size, self.current)

    def get_last_batch_basename_list(self):
        assert self.fixed_order
        l, r = self.get_last_batch_range()
        return self.basename_list[l:r]

    def create_iterator(self):
        return self
