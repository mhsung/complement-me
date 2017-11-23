#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Minhyuk Sung (mhsung@cs.stanford.edu)
# April 2017

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

from global_variables import *
from dataset import Dataset
from mixture_density_network import MixtureDensityNetwork
from mixture_density_sample import *
from train_util import evaluate, train
import gflags
import numpy as np
import random
import tensorflow as tf


FLAGS = gflags.FLAGS
gflags.DEFINE_string('in_model_dirs', '', '')
gflags.DEFINE_string('in_model_scopes', '', '')
gflags.DEFINE_string('out_model_dir', 'model', '')
gflags.DEFINE_string('out_dir', 'outputs', '')
gflags.DEFINE_string('log_dir', 'log', '')

gflags.DEFINE_string('loss_func', '',\
        'joint_embedding, X_embedding, Y_embedding, and position')

gflags.DEFINE_bool('train', False, '')
gflags.DEFINE_string('optimizer', 'adam',\
        'adam or momentum [default: adam]')
gflags.DEFINE_float('init_learning_rate', 0.001,\
        'Initial learning rate [default: 0.001]')
gflags.DEFINE_float('momentum', 0.9,\
        'Initial learning rate [default: 0.9]')
gflags.DEFINE_float('decay_step', 50000,\
        'Decay step for lr decay [default: 50000]')
gflags.DEFINE_float('decay_rate', 0.8,\
        'Decay rate for lr decay [default: 0.8]')

gflags.DEFINE_integer('K', 8, 'Number of Gaussian modes. Run regression if 0.')
gflags.DEFINE_integer('D', 50, '')
gflags.DEFINE_float('max_sigma', 0.05, '')

gflags.DEFINE_integer('n_epochs', 2000, '')
gflags.DEFINE_integer('batch_size', 32, '')
gflags.DEFINE_integer('snapshot_epoch', 100, '')

gflags.DEFINE_bool('test_single_target', False, '')
gflags.DEFINE_bool('test_single_given', False, '')


def print_params():
    print('==== PARAMS ====')
    if FLAGS.K == 0:
        raise AssertionError
    else:
        print(' - MDN (K = {:d}, D = {:d})'.format(FLAGS.K, FLAGS.D))
    print(' - Sphere Embedding: {}'.format(FLAGS.sphere_embedding))


def load_model(sess, in_model_dir, exclude=''):
    # Read variables names in checkpoint.
    var_names = [x for x,_ in tf.contrib.framework.list_variables(in_model_dir)]

    # Find variables with given names.
    # HACK:
    # Convert unicode to string and remove postfix ':0'.
    var_list = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)\
            if str(x.name)[:-2] in var_names]

    if exclude != '':
        var_list = [x for x in var_list if exclude not in x.name]
    #print([x.name for x in var_list])

    saver = tf.train.Saver(var_list)

    ckpt = tf.train.get_checkpoint_state(in_model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ("Loaded '{}'.".format(ckpt.model_checkpoint_path))
    else:
        print ("Failed to loaded '{}'.".format(in_model_dir))
        return False
    return True


if __name__ == '__main__':
    FLAGS(sys.argv)
    tf.set_random_seed(0)
    np.random.seed(0)
    random.seed(0)

    train_data = Dataset(g_group_train_pairs_file, FLAGS.batch_size,
            FLAGS.D, single_target=False)
    test_data = Dataset(g_group_test_pairs_file, FLAGS.batch_size, FLAGS.D,
            single_target=FLAGS.test_single_target,
            single_given=FLAGS.test_single_given)

    print_params()

    net = MixtureDensityNetwork(train_data.n_points, FLAGS.D,
            FLAGS.K, FLAGS.max_sigma, FLAGS.batch_size, FLAGS.optimizer,
            FLAGS.init_learning_rate, FLAGS.momentum, FLAGS.decay_step,
            FLAGS.decay_rate, FLAGS.loss_func, FLAGS.sphere_embedding)


    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config, graph=net.graph) as sess:
        sess.run(tf.global_variables_initializer(), {net.is_training: True})

        if FLAGS.in_model_dirs:
            exclude = ''
            if 'embedding' in FLAGS.loss_func:
                exclude = 'Yp'
            elif 'position' in FLAGS.loss_func:
                exclude = 'Yc'

            for in_model_dir in FLAGS.in_model_dirs.split(','):
                assert(load_model(sess, in_model_dir, exclude))

        if FLAGS.train:
            train(sess, net, train_data, test_data, n_epochs=FLAGS.n_epochs,
                    snapshot_epoch=FLAGS.snapshot_epoch,
                    model_dir=FLAGS.out_model_dir, log_dir=FLAGS.log_dir,
                    data_name=g_shape_synset, output_generator=None)
        else:
            '''
            train_loss, train_accuracy, _ = evaluate(sess, net, train_data)
            test_loss, test_accuracy, _ = evaluate(sess, net, test_data)

            msg = "|| Train Loss: {:6f}".format(train_loss)
            msg += " | Train Accu: {:5f}".format(train_accuracy)
            msg += " | Test Loss: {:6f}".format(test_loss)
            msg += " | Test Accu: {:5f}".format(test_accuracy)
            msg += " ||"
            print(msg)
            '''

            if 'joint_embedding' in FLAGS.loss_func or\
                    FLAGS.loss_func == 'X_embedding':
                generate_embedding_outputs(sess, net, test_data, FLAGS.out_dir,
                        FLAGS.loss_func)
            elif FLAGS.loss_func == 'position':
                generate_position_outputs(sess, net, test_data, FLAGS.out_dir)
            else:
                raise AssertionError

