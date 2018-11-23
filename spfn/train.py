import os, sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, 'lib'))
sys.path.append(os.path.join(BASE_DIR, '..'))

from network_config import NetworkConfig
from network import Network
from dataset import Dataset
from utils.differentiable_tls import register_custom_svd_gradient
import fitter_factory

import random
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import os
import json
import argparse

if __name__ == '__main__':
    tf.set_random_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    register_custom_svd_gradient()

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='YAML configuration file')
    parser.add_argument('--test', action='store_true', help='Run network in test time')
    parser.add_argument('--test_pc_in', type=str)
    parser.add_argument('--test_h5_out', type=str)
    args = parser.parse_args()

    conf = NetworkConfig(args.config_file)

    visible_GPUs = conf.get_CUDA_visible_GPUs()
    if visible_GPUs is not None:
        print('Setting CUDA_VISIBLE_DEVICES={}'.format(','.join(visible_GPUs)))
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(visible_GPUs)

    fitter_factory.register_primitives(conf.get_list_of_primitives())

    is_testing = True if args.test else False
    batch_size = conf.get_batch_size()
    n_max_instances = conf.get_n_max_instances()

    tf_conf = tf.ConfigProto()
    tf_conf.allow_soft_placement = True
    tf_conf.gpu_options.allow_growth = True

    in_model_dir = conf.get_in_model_dir()
    ckpt = tf.train.get_checkpoint_state(in_model_dir)
    should_restore = (ckpt is not None) and (ckpt.model_checkpoint_path is not None)

    print('Building network...')
    net = Network(n_max_instances=n_max_instances, config=conf, is_new_training=not should_restore)
    with tf.Session(config=tf_conf, graph=net.graph) as sess:
        if conf.is_debug_mode():
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        if should_restore:
            print('Restoring ' + ckpt.model_checkpoint_path + ' ...')
            tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
        else:
            assert not is_testing
            print('Starting a new training...')
            sess.run(tf.global_variables_initializer())

        print('Loading data...')
        if is_testing:
            if args.test_pc_in is not None:
                # single point cloud testing
                assert args.test_h5_out is not None
                net.simple_predict_and_save(
                    sess,
                    pc=np.genfromtxt(args.test_pc_in, delimiter=' ', dtype=float)[:, :3],
                    pred_h5_file=args.test_h5_out
                )
            else:
                # batch testing
                test_data = Dataset(
                    batch_size=batch_size, 
                    n_max_instances=n_max_instances, 
                    csv_path=conf.get_test_data_file(), 
                    noisy=conf.is_test_data_noisy(), 
                    fixed_order=True, 
                    first_n=conf.get_test_data_first_n()
                )
                net.predict_and_save(
                    sess,
                    dset=test_data,
                    save_dir=conf.get_test_prediction_dir(),
                )
        else:
            train_data = Dataset(
                batch_size=batch_size, 
                n_max_instances=n_max_instances, 
                csv_path=conf.get_train_data_file(), 
                noisy=conf.is_train_data_noisy(), 
                fixed_order=False, 
                first_n=conf.get_train_data_first_n()
            )
            val_data = Dataset(
                batch_size=batch_size, 
                n_max_instances=n_max_instances, 
                csv_path=conf.get_val_data_file(), 
                noisy=conf.is_val_data_noisy(), 
                fixed_order=True, 
                first_n=conf.get_val_data_first_n()
            )
            net.train(
                sess, 
                train_data=train_data, 
                val_data=val_data, 
                n_epochs=conf.get_n_epochs(), 
                val_interval=conf.get_val_interval(),
                snapshot_interval=conf.get_snapshot_interval(),
                model_dir=conf.get_out_model_dir(),
                log_dir=conf.get_log_dir(),
            )

