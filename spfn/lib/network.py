import os, sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

import architecture
import evaluation
import fitter_factory
import prediction_io

import time
import numpy as np
import tensorflow as tf
import re
import subprocess

class Network(object):
    def __init__(self, n_max_instances, config, is_new_training):
        self.n_max_instances = n_max_instances
        self.config = config

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.Variable(0)

            self.is_training = tf.placeholder(dtype=tf.bool, shape=[])
            self.P = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
            self.batch_size = tf.shape(self.P)[0]

            if config.get_bn_decay_step() < 0:
                self.bn_decay = None
            else:
                self.bn_decay = self.get_batch_norm_decay(self.global_step, self.batch_size, config.get_bn_decay_step())
                tf.summary.scalar('bn_decay', self.bn_decay)

            self.list_of_primitives = config.get_list_of_primitives()
            self.gt_dict = evaluation.create_gt_dict(n_max_instances)

            if config.use_direct_regression():
                self.pred_dict, direct_loss = architecture.get_direct_regression_model(
                    scope='DPPN', 
                    P=self.P, 
                    n_max_instances=n_max_instances, 
                    gt_dict=self.gt_dict,
                    is_training=self.is_training, 
                    bn_decay=self.bn_decay
                )
                self.total_loss = tf.reduce_mean(direct_loss)
                self.total_miou_loss = tf.zeros(shape=[], dtype=tf.float32)
                self.total_normal_loss = tf.zeros(shape=[], dtype=tf.float32)
                self.total_type_loss = tf.zeros(shape=[], dtype=tf.float32)
                self.total_residue_loss = tf.zeros(shape=[], dtype=tf.float32)
                self.total_parameter_loss = tf.zeros(shape=[], dtype=tf.float32)
            else:
                self.pred_dict = architecture.get_per_point_model(
                    scope='SPFN', 
                    P=self.P, 
                    n_max_instances=n_max_instances, 
                    is_training=self.is_training, 
                    bn_decay=self.bn_decay,
                )

                eval_dict = evaluation.evaluate(
                    self.pred_dict, 
                    self.gt_dict, 
                    is_eval=False,
                    is_nn=True
                )
                self.collect_losses(eval_dict['loss_dict'])

            learning_rate = self.get_learning_rate(
                config.get_init_learning_rate(),
                self.global_step,
                self.batch_size,
                config.get_decay_step(),
                config.get_decay_rate())
            tf.summary.scalar('learning_rate', learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.create_train_op(learning_rate, self.total_loss)

            self.summary = tf.summary.merge_all()
            self.saver = tf.train.Saver(max_to_keep=3)

    def create_train_op(self, learning_rate, total_loss):
        # Skip gradient update if any gradient is infinite. This should not happen and is for debug only.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.optimizer = optimizer
        grads_and_vars = optimizer.compute_gradients(total_loss)
        grads = [g for g, v in grads_and_vars]
        varnames = [v for g, v in grads_and_vars]
        is_finite = tf.ones(dtype=tf.bool, shape=[])
        for g, v in grads_and_vars:
            if g is not None:
                g_is_finite = tf.reduce_any(tf.is_finite(g))
                g_is_finite_cond = tf.cond(g_is_finite, tf.no_op, lambda: tf.Print(g_is_finite, [g], '{} is not finite:'.format(str(g))))
                with tf.control_dependencies([g_is_finite_cond]):
                    is_finite = tf.logical_and(is_finite, g_is_finite)
        train_op = tf.cond(
            is_finite, 
            lambda: optimizer.apply_gradients(zip(grads, varnames), global_step=self.global_step), 
            lambda: tf.Print(is_finite, [is_finite], 'Some gradients are not finite! Skipping gradient backprop.')
        )
        return train_op

    def collect_losses(self, loss_dict):
        self.total_loss = tf.zeros(shape=[], dtype=tf.float32)

        self.normal_loss_per_data = loss_dict['normal_loss']
        self.total_normal_loss = tf.reduce_mean(self.normal_loss_per_data)
        normal_loss_multiplier = self.config.get_normal_loss_multiplier()
        if normal_loss_multiplier > 0:
            tf.summary.scalar('total_normal_loss', self.total_normal_loss)
            self.total_loss += normal_loss_multiplier * self.total_normal_loss

        self.type_loss_per_data = loss_dict['type_loss']
        self.total_type_loss = tf.reduce_mean(self.type_loss_per_data)
        type_loss_multiplier = self.config.get_type_loss_multiplier()
        if type_loss_multiplier > 0:
            tf.summary.scalar('total_type_loss', self.total_type_loss)
            self.total_loss += type_loss_multiplier * self.total_type_loss

        self.miou_loss_per_data = loss_dict['avg_miou_loss']
        self.miou_loss_per_instance = loss_dict['miou_loss']
        self.total_miou_loss = tf.reduce_mean(self.miou_loss_per_data)
        miou_loss_multiplier = self.config.get_miou_loss_multiplier()
        if miou_loss_multiplier > 0:
            tf.summary.scalar('total_miou_loss', self.total_miou_loss)
            self.total_loss += miou_loss_multiplier * self.total_miou_loss

        self.residue_loss_per_data = loss_dict['avg_residue_loss']
        self.residue_loss_per_instance = loss_dict['residue_loss']
        self.total_residue_loss = tf.reduce_mean(self.residue_loss_per_data)
        residue_loss_multiplier = self.config.get_residue_loss_multiplier()
        if residue_loss_multiplier > 0:
            tf.summary.scalar('total_residue_loss', self.total_residue_loss)
            self.total_loss += residue_loss_multiplier * self.total_residue_loss

        self.parameter_loss_per_data = loss_dict['avg_parameter_loss']
        self.parameter_loss_per_instance = loss_dict['parameter_loss']
        self.total_parameter_loss = tf.reduce_mean(self.parameter_loss_per_data)
        parameter_loss_multiplier = self.config.get_parameter_loss_multiplier()
        if parameter_loss_multiplier > 0:
            tf.summary.scalar('total_parameter_loss', self.total_parameter_loss)
            self.total_loss += parameter_loss_multiplier * self.total_parameter_loss

        self.total_loss *= self.config.get_total_loss_multiplier()
        tf.summary.scalar('total_loss', self.total_loss)

    def train(self, sess, train_data, val_data, n_epochs, val_interval, snapshot_interval, model_dir, log_dir):
        assert n_epochs > 0

        train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(log_dir, 'val'), sess.graph)
        if not os.path.exists(model_dir): 
            os.makedirs(model_dir)
        if not os.path.exists(self.config.get_val_prediction_dir()):
            os.makedirs(self.config.get_val_prediction_dir())
        print('Training started.')

        start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            for batch in train_data.create_iterator():
                feed_dict = self.create_feed_dict(batch, is_training=True)
                step, _, summary, loss = sess.run([self.global_step, self.train_op, self.summary, self.total_loss], feed_dict=feed_dict)

                elapsed_min = (time.time() - start_time) / 60
                print('Epoch: {:d} | Step: {:d} | Batch Loss: {:6f} | Elapsed: {:.2f}m'.format(epoch, step, loss, elapsed_min))

                if step >= self.config.get_writer_start_step():
                    train_writer.add_summary(summary, step)
            
                if step % val_interval == 0:
                    print('Start validating...')
                    msg = 'Epoch: {:d} | Step: {:d}'.format(epoch, step)

                    remain_min = (n_epochs * train_data.n_data - step) * elapsed_min / step

                    predict_result = self.predict_and_save(sess, val_data, save_dir=os.path.join(self.config.get_val_prediction_dir(), 'step{}'.format(step)))
                    msg = predict_result['msg']
                    msg = 'Validation: ' + msg + ' | Elapsed: {:.2f}m, Remaining: {:.2f}m'.format(elapsed_min, remain_min)
                    print(msg)
                    # clean up old predictions
                    prediction_n_keep = self.config.get_val_prediction_n_keep()
                    if prediction_n_keep != -1:
                        self.clean_predictions_earlier_than(step=step, prediction_dir=self.config.get_val_prediction_dir(), n_keep=prediction_n_keep)
                    if step >= self.config.get_writer_start_step():
                        val_writer.add_summary(predict_result['summary'], step)
                
                if step % snapshot_interval == 0:
                    print('Saving snapshot at step {:d}...'.format(step))
                    self.saver.save(sess, os.path.join(model_dir, 'tf_model.ckpt'), global_step=step)
                    print('Done saving model at step {:d}.'.format(step))

        train_writer.close()
        val_writer.close();
        elapsed_min = (time.time() - start_time) / 60
        print('Training finished.')
        print('Elapsed: {:.2f}m.'.format(elapsed_min))
        print('Saved {}.'.format(self.saver.save(sess, os.path.join(model_dir, 'tf_model.ckpt'), global_step=step)))

    def format_loss_result(self, losses):
        msg = ''
        msg += 'Total Loss: {:6f}'.format(losses['total_loss'])
        msg += ', MIoU Loss: {:6f}'.format(losses['total_miou_loss'])
        msg += ', Normal Loss: {:6f}'.format(losses['total_normal_loss'])
        msg += ', Type Loss: {:6f}'.format(losses['total_type_loss'])
        msg += ', Parameter Loss: {:6f}'.format(losses['total_parameter_loss'])
        msg += ', Residue Loss: {:6f}'.format(losses['total_residue_loss'])
        return msg

    def clean_predictions_earlier_than(self, step, prediction_dir, n_keep):
        prog = re.compile('step([0-9]+)')
        arr = []
        for f in os.listdir(prediction_dir):
            if os.path.isdir(os.path.join(prediction_dir, f)):
                m = prog.match(f)
                if m is not None:
                    arr.append((int(m.group(1)), f))
        arr.sort(key=lambda pr: pr[0])
        for pr in arr[:-n_keep]:
            subprocess.run(['rm', '-r', os.path.join(prediction_dir, pr[1])])

    def predict_and_save(self, sess, dset, save_dir):
        print('Predicting and saving predictions to {}...'.format(save_dir))
        losses = {
            'total_loss': 0.0, 
            'total_miou_loss': 0.0, 
            'total_normal_loss': 0.0,
            'total_type_loss': 0.0,
            'total_residue_loss': 0.0, 
            'total_parameter_loss': 0.0
        }
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for batch in dset.create_iterator():
            feed_dict = self.create_feed_dict(batch, is_training=False)
            loss_dict = {
                'total_loss': self.total_loss,
                'total_miou_loss': self.total_miou_loss,
                'total_normal_loss': self.total_normal_loss,
                'total_type_loss': self.total_type_loss,
                'total_residue_loss': self.total_residue_loss,
                'total_parameter_loss': self.total_parameter_loss,
            }
            pred_result, loss_result = sess.run([self.pred_dict, loss_dict], feed_dict=feed_dict)

            for key in losses.keys():
                losses[key] += loss_result[key] * dset.last_step_size
            prediction_io.save_batch_nn(
                nn_name=self.config.get_nn_name(),
                pred_result=pred_result, 
                basename_list=dset.get_last_batch_basename_list(), 
                save_dir=save_dir,
                W_reduced=False,
            )
            print('Finished {}/{}'.format(dset.get_last_batch_range()[1], dset.n_data), end='\r')
        losses.update((x, y / dset.n_data) for x, y in losses.items())
        msg = self.format_loss_result(losses)
        open(os.path.join(save_dir, 'test_loss.txt'), 'w').write(msg)
        summary = tf.Summary()
        for x, y in losses.items():
            summary.value.add(tag=x, simple_value=y)
        return {
            'msg': msg,
            'summary': summary,
        }

    def simple_predict_and_save(self, sess, pc, pred_h5_file):
        feed_dict = {
            self.P: np.expand_dims(pc, axis=0), # 1xNx3
            self.is_training: False
        }
        pred_result = sess.run(self.pred_dict, feed_dict=feed_dict)
        prediction_io.save_single_nn(
            nn_name=self.config.get_nn_name(),
            pred_result=pred_result, 
            pred_h5_file=pred_h5_file,
            W_reduced=False,
        )
        
    def create_feed_dict(self, batch, is_training):
        feed_dict = {
            self.P : batch['P'], 
            self.is_training: is_training,
        }
        evaluation.fill_gt_dict_with_batch_data(feed_dict, self.gt_dict, batch)
        return feed_dict
        
    def get_batch_norm_decay(self, global_step, batch_size, bn_decay_step):
        BN_INIT_DECAY = 0.5
        BN_DECAY_RATE = 0.5
        BN_DECAY_CLIP = 0.99

        bn_momentum = tf.train.exponential_decay(
            BN_INIT_DECAY,
            global_step*batch_size,
            bn_decay_step,
            BN_DECAY_RATE,
            staircase=True)

        bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay

    def get_learning_rate(self, init_learning_rate, global_step, batch_size, decay_step, decay_rate):
        learning_rate = tf.train.exponential_decay(
            init_learning_rate,
            global_step*batch_size,
            decay_step,
            decay_rate,
            staircase=True)
        return learning_rate


