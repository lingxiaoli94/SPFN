import yaml

class EvalConfig(object):
    def __init__(self, filename):
        self.conf = yaml.load(open(filename, 'r'))

    def fetch(self, name, default_value=None):
        result = self.conf.get(name, default_value)
        assert result is not None
        return result

    def get_batch_size(self):
        return self.fetch('batch_size')

    def get_test_data_file(self):
        return self.fetch('test_data_file')

    def get_test_data_first_n(self):
        return self.fetch('test_first_n')

    def is_test_data_noisy(self):
        return self.fetch('test_data_noisy')

    def get_prediction_csv_file(self):
        return self.fetch('prediction_csv_file')

    def get_bundle_dir(self):
        return self.fetch('bundle_dir')

    def get_n_max_instances(self):
        return self.fetch('n_max_instances')

    def get_list_of_primitives(self):
        return self.fetch('list_of_primitives')

    def get_CUDA_visible_GPUs(self):
        return self.fetch('CUDA_visible_GPUs')

    def is_nn(self):
        return self.fetch('is_nn')
