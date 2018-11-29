import yaml

class NetworkConfig(object):
    def __init__(self, filename):
        self.conf = yaml.load(open(filename, 'r'))

    def fetch(self, name, default_value=None):
        result = self.conf.get(name, default_value)
        assert result is not None
        return result

    def get_nn_name(self):
        return self.fetch('nn_name')

    def get_batch_size(self):
        return self.fetch('batch_size')

    def get_total_loss_multiplier(self):
        return self.fetch('total_loss_multiplier')

    def get_normal_loss_multiplier(self):
        return self.fetch('normal_loss_multiplier')

    def get_type_loss_multiplier(self):
        return self.fetch('type_loss_multiplier')

    def get_residue_loss_multiplier(self):
        return self.fetch('residue_loss_multiplier')

    def get_parameter_loss_multiplier(self):
        return self.fetch('parameter_loss_multiplier')

    def get_miou_loss_multiplier(self):
        return self.fetch('miou_loss_multiplier')

    def get_bn_decay_step(self):
        return self.fetch('bn_decay_step', -1)

    def get_init_learning_rate(self):
        return self.fetch('init_learning_rate')

    def get_decay_step(self):
        return self.fetch('decay_step')

    def get_decay_rate(self):
        return self.fetch('decay_rate')

    def get_n_epochs(self):
        return self.fetch('n_epochs')

    def get_val_interval(self):
        return self.fetch('val_interval', 5)
    
    def get_snapshot_interval(self):
        return self.fetch('snapshot_interval', 100)

    def get_in_model_dir(self):
        return self.fetch('in_model_dir')

    def get_out_model_dir(self):
        return self.fetch('out_model_dir')

    def get_log_dir(self):
        return self.fetch('log_dir')

    def get_train_data_file(self):
        return self.fetch('train_data_file')

    def get_train_data_first_n(self):
        return self.fetch('train_first_n')

    def is_train_data_noisy(self):
        return self.fetch('train_data_noisy')

    def get_val_data_file(self):
        return self.fetch('val_data_file')

    def get_val_data_first_n(self):
        return self.fetch('val_first_n')

    def is_val_data_noisy(self):
        return self.fetch('val_data_noisy')

    def get_val_prediction_dir(self):
        return self.fetch('val_prediction_dir')

    def get_val_prediction_n_keep(self):
        return self.fetch('val_prediction_n_keep')

    def get_test_data_file(self):
        return self.fetch('test_data_file')

    def get_test_data_first_n(self):
        return self.fetch('test_first_n')

    def is_test_data_noisy(self):
        return self.fetch('test_data_noisy')

    def get_test_prediction_dir(self):
        return self.fetch('test_prediction_dir')

    def get_CUDA_visible_GPUs(self):
        return self.fetch('CUDA_visible_GPUs')

    def get_writer_start_step(self):
        return self.fetch('writer_start_step')

    def is_debug_mode(self):
        return self.fetch('debug_mode')

    def get_n_max_instances(self):
        return self.fetch('n_max_instances')

    def get_list_of_primitives(self):
        return self.fetch('list_of_primitives')

    def use_direct_regression(self):
        return self.fetch('use_direct_regression')
