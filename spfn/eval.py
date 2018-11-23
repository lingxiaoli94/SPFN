import os, sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, 'lib'))
sys.path.append(os.path.join(BASE_DIR, '..'))

import numpy as np
import argparse
import tensorflow as tf

from dataset import Dataset
from eval_config import EvalConfig
import fitter_factory
from prediction_io import PredictionLoader
import evaluation
import bundle_io

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='YAML configuration file')
    args = parser.parse_args()

    conf = EvalConfig(args.config_file)

    visible_GPUs = conf.get_CUDA_visible_GPUs()
    if visible_GPUs is not None:
        print('Setting CUDA_VISIBLE_DEVICES={}'.format(','.join(visible_GPUs)))
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(visible_GPUs)

    fitter_factory.register_primitives(conf.get_list_of_primitives())

    n_max_instances = conf.get_n_max_instances()
    batch_size = conf.get_batch_size();
    test_data = Dataset(
        batch_size=batch_size, 
        n_max_instances=n_max_instances, 
        csv_path=conf.get_test_data_file(), 
        noisy=conf.is_test_data_noisy(), 
        first_n=conf.get_test_data_first_n(), 
        fixed_order=True
    )
    pred_loader = PredictionLoader(n_max_instances=n_max_instances, csv_path=conf.get_prediction_csv_file())
    bundle_dir = conf.get_bundle_dir()
    if not os.path.exists(bundle_dir):
        os.makedirs(bundle_dir)

    tf_conf = tf.ConfigProto()
    tf_conf.allow_soft_placement = True
    tf_conf.gpu_options.allow_growth = True

    graph = tf.Graph()
    with graph.as_default():
        pred_ph = {}
        pred_ph['W'] = tf.placeholder(shape=[None, None, n_max_instances], dtype=tf.float32)
        pred_ph['normal_per_point'] = tf.placeholder(shape=[None, None, 3], dtype=tf.float32)
        pred_ph['type_per_point'] = tf.placeholder(shape=[None, None], dtype=tf.int32) # should be BxN in test
        pred_ph['parameters'] = {}
        for fitter_cls in fitter_factory.get_all_fitter_classes():
            fitter_cls.insert_prediction_placeholders(pred_ph['parameters'], n_max_instances)

        gt_ph = evaluation.create_gt_dict(n_max_instances)
        P_in = tf.placeholder(shape=[None, None, 3], dtype=tf.float32)
        eval_result_node = evaluation.evaluate(pred_ph, gt_ph, is_eval=True, is_nn=conf.is_nn(), P_in=P_in)

    stats = {
        'total_miou_loss': 0.0, 
        'total_normal_loss': 0.0,
        'total_type_loss': 0.0,
        'total_residue_loss': 0.0, 
        'total_parameter_loss': 0.0,
        'per_instance_type_accuracy': 0.0,
        'avg_residue_loss_without_gt': 0.0,
        'parameter_loss_without_gt': 0.0,
    }
    # Finish building evaluation graph. Start to run evaluations...
    with tf.Session(config=tf_conf, graph=graph) as sess:
        for batch in test_data.create_iterator():
            feed_dict = {}
            feed_dict[P_in] = batch['P']

            evaluation.fill_gt_dict_with_batch_data(feed_dict, gt_ph, batch) 
            pred_result, method_name_list = pred_loader.load_multiple(test_data.get_last_batch_basename_list())

            assert len(pred_result['instance_per_point'].shape) == 3 # BxNxK
            feed_dict[pred_ph['W']] = pred_result['instance_per_point']
            feed_dict[pred_ph['normal_per_point']] = pred_result['normal_per_point']
            feed_dict[pred_ph['type_per_point']] = pred_result['type_per_point']

            for param in pred_result['parameters'].keys():
                feed_dict[pred_ph['parameters'][param]] = pred_result['parameters'][param]

            eval_result = sess.run(eval_result_node, feed_dict=feed_dict) # {loss_dict, matching_indices}
            stats['total_miou_loss'] += np.sum(eval_result['loss_dict']['avg_miou_loss'])
            stats['total_normal_loss'] += np.sum(eval_result['loss_dict']['normal_loss'])
            stats['total_type_loss'] += np.sum(eval_result['loss_dict']['type_loss'])
            stats['total_residue_loss'] += np.sum(eval_result['loss_dict']['avg_residue_loss'])
            stats['total_parameter_loss'] += np.sum(eval_result['loss_dict']['avg_parameter_loss'])
            stats['per_instance_type_accuracy'] += np.sum(eval_result['stats']['per_instance_type_accuracy'])
            stats['avg_residue_loss_without_gt'] += np.sum(eval_result['stats']['avg_residue_loss_without_gt'])
            stats['parameter_loss_without_gt'] += np.sum(eval_result['stats']['parameter_loss_without_gt'])

            print('miou loss: {}'.format(eval_result['loss_dict']['avg_miou_loss']))
            print('Stats: {}'.format(eval_result['stats']))
            batch_size = batch['P'].shape[0]
            basename_list = test_data.get_last_batch_basename_list()
            for b in range(batch_size):
                data_current = {key: batch[key][b] for key in batch.keys()}
                loss_current = {key: eval_result['loss_dict'][key][b] for key in eval_result['loss_dict'].keys()}
                stats_current = {key: eval_result['stats'][key][b] for key in eval_result['stats'].keys()}
                pred_current = {}
                for key in pred_result.keys():
                    if key != 'parameters':
                        pred_current[key] = pred_result[key][b]
                    else:
                        pred_current['parameters'] = {}
                        for key2 in pred_result['parameters'].keys():
                            pred_current['parameters'][key2] = pred_result['parameters'][key2][b]
                bundle_io.dump_single_bundle(
                    data=data_current, 
                    pred_result=pred_current, 
                    loss_dict=loss_current,
                    stats=stats_current,
                    matching_indices=eval_result['matching_indices'][b], # K
                    instance_per_point=eval_result['instance_per_point'][b], # N
                    type_per_instance=eval_result['type_per_instance'][b], # K
                    null_mask=eval_result['null_mask'][b], # K
                    mask_gt_nulled=eval_result['mask_gt_nulled'][b], # K
                    residue_to_closest=eval_result['residue_to_closest'][b], # N
                    residue_gt_primitive=eval_result['residue_gt_primitive'][b], # KxN'
                    list_of_primitives=conf.get_list_of_primitives(),
                    method_name=method_name_list[b],
                    output_prefix=os.path.join(bundle_dir, basename_list[b]),
                )

    stats.update((x, y / test_data.n_data) for x, y in stats.items())
    open(os.path.join(bundle_dir, 'eval_stats_avg.txt'), 'w').write('\n'.join(['{}: {}'.format(x, y) for x, y in stats.items()]))
