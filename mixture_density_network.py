# -*- coding: utf-8 -*-
# Minhyuk Sung (mhsung@cs.stanford.edu)
# April 2017

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from dataset import Dataset
import math
import numpy as np
import tensorflow as tf
import tf_util


class MixtureDensityNetwork(object):
    def __init__(self, n_points, Yc_dim, K, max_sigma, batch_size, optimizer,
            init_learning_rate, momentum, decay_step, decay_rate, loss_func):
        self.n_points = n_points
        self.Yc_dim = Yc_dim
        self.K = K
        self.max_sigma = max_sigma
        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            self.global_step = tf.Variable(0)

            self.bn_decay = self.get_batch_norm_decay()
            tf.summary.scalar('bn_decay', self.bn_decay)

            self.is_training = tf.placeholder(tf.bool, shape=())

            # Build network.
            self.build_nets(self.is_training, self.bn_decay)

            X_var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                    scope='X')
            Y_var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                    scope='Y')

            # Define loss, accuracy, and variables to be saved.
            if loss_func == 'joint_embedding':
                self.create_embedding_loss_and_accuracy(
                        self.pred_Yc, self.pred_Zc)
            elif loss_func == 'position':
                self.create_position_loss_and_accuracy()
            else:
                raise AssertionError

            self.learning_rate = self.get_learning_rate()
            tf.summary.scalar('learning_rate', self.learning_rate)

            if optimizer == 'momentum':
                self.train_op = tf.train.MomentumOptimizer(
                        self.learning_rate, momentum=momentum).minimize(
                                self.loss, global_step=self.global_step)
            elif optimizer == 'adam':
                self.train_op = tf.train.AdamOptimizer(
                        self.learning_rate).minimize(
                                self.loss, global_step=self.global_step)
            else:
                raise AssertionError

            # Define merged summary.
            self.summary = tf.summary.merge_all()

            # Define saver.
            self.saver = tf.train.Saver(max_to_keep=0)


    def build_nets(self, is_training, bn_decay):
        # FIXME:
        # Make the placeholders to have dynamic sizes.

        # X: Partial input point cloud.
        self.X = tf.placeholder(tf.float32,
                shape=[self.batch_size, self.n_points, 3])

        # Y: Positive sample.
        # Point cloud.
        self.Y = tf.placeholder(tf.float32,
                shape=[self.batch_size, self.n_points, 3])
        # Position.
        self.Yp = tf.placeholder(tf.float32,
                shape=[self.batch_size, 3])

        # Z: Negative sample.
        # Point cloud.
        self.Z = tf.placeholder(tf.float32,
                shape=[self.batch_size, self.n_points, 3])

        scope = 'X'
        reuse = False
        with tf.variable_scope(scope, reuse=reuse) as sc:
            self.X_net, self.X_feature = self.build_pointnet(
                    self.X, is_training, bn_decay, scope, reuse)

        scope = 'Y'
        reuse = False
        with tf.variable_scope(scope, reuse=reuse) as sc:
            self.Y_net, self.Y_feature = self.build_pointnet(
                    self.Y, is_training, bn_decay, scope, reuse)

        scope = 'Pred_Yc'
        reuse = False
        with tf.variable_scope(scope, reuse=reuse) as sc:
            self.pred_Yc = tf_util.fully_connected(self.Y_net, self.Yc_dim,
                    activation_fn=None, scope=scope+'_pred_Yc', reuse=reuse)

        # NOTE:
        # Share the same weights for Y and Z.
        scope = 'Y'
        reuse = True
        with tf.variable_scope(scope, reuse=reuse) as sc:
            self.Z_net, self.Z_feature = self.build_pointnet(
                    self.Z, is_training, bn_decay, scope, reuse)

        scope = 'Pred_Yc'
        reuse = True
        with tf.variable_scope(scope, reuse=reuse) as sc:
            self.pred_Zc = tf_util.fully_connected(self.Z_net, self.Yc_dim,
                    activation_fn=None, scope=scope+'_pred_Yc', reuse=reuse)

        # Define X MDN params.
        scope = 'MDN_X'
        reuse = False
        with tf.variable_scope(scope, reuse=reuse) as sc:
            self.log_logits, self.mus, self.sigmas =\
                    self.build_mixture_density_params(
                            self.X_net, self.max_sigma, scope, reuse)
            self.logits = tf.exp(self.log_logits)

        # Define Y predicted position.
        scope = 'Pred_Yp'
        reuse = False
        with tf.variable_scope(scope, reuse=reuse) as sc:
            net = tf.concat([self.X_feature, self.Y_feature], 1)

            net = tf_util.fully_connected(net, 512, bn=True,
                    is_training=is_training, bn_decay=bn_decay,
                    scope=scope+'_fc1', reuse=reuse)

            net = tf_util.fully_connected(net, 256, bn=True,
                    is_training=is_training, bn_decay=bn_decay,
                    scope=scope+'_fc2', reuse=reuse)

            net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                    scope=scope+'_dp1', reuse=reuse)

            self.pred_Yp = tf_util.fully_connected(net, 3,
                    activation_fn=None, scope=scope+'_pred_Yp', reuse=reuse)


    def create_embedding_loss_and_accuracy(self, input_Yc, input_Zc):
        with tf.name_scope('loss'):
            self.Y_losses, self.Y_probs = self.compute_gaussian_mixture_loss(
                    input_Yc, self.log_logits, self.mus, self.sigmas)
            self.Z_losses, _ = self.compute_gaussian_mixture_loss(
                    input_Zc, self.log_logits, self.mus, self.sigmas)

            m = 10
            # http://www.jmlr.org/papers/volume11/chechik10a/chechik10a.pdf
            self.combined_losses = tf.nn.relu(
                    self.Y_losses - self.Z_losses + m)
            self.loss = tf.reduce_mean(self.combined_losses)
        tf.summary.scalar('loss', self.loss)

        self.dist_tol = 0.1
        with tf.name_scope('accuracy'):
            self.dists = self.compute_min_dists_to_means(
                    input_Yc, self.mus)
            self.correct_pred = tf.less_equal(self.dists, self.dist_tol)
            self.accuracy = tf.reduce_mean(
                    tf.cast(self.correct_pred, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)


    def create_position_loss_and_accuracy(self):
        with tf.name_scope('loss'):
            self.losses = tf.reduce_sum(tf.square(self.pred_Yp - self.Yp), 1)
            self.loss = tf.reduce_mean(self.losses)
        tf.summary.scalar('loss', self.loss)

        self.dist_tol = 0.01
        with tf.name_scope('accuracy'):
            self.dists = tf.sqrt(self.losses)
            self.correct_pred = tf.less_equal(self.dists, self.dist_tol)
            self.accuracy = tf.reduce_mean(
                    tf.cast(self.correct_pred, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)


    def get_learning_rate(self):
        learning_rate = tf.train.exponential_decay(
                            self.init_learning_rate,
                            self.global_step*self.batch_size,
                            self.decay_step,
                            self.decay_rate,
                            staircase=True)
        learing_rate = tf.maximum(learning_rate, 0.00001)
        return learning_rate


    def get_batch_norm_decay(self):
        BN_INIT_DECAY = 0.5
        BN_DECAY_RATE = 0.5
        BN_DECAY_CLIP = 0.99

        bn_momentum = tf.train.exponential_decay(
                        BN_INIT_DECAY,
                        self.global_step*self.batch_size,
                        self.decay_step,
                        BN_DECAY_RATE,
                        staircase=True)

        bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay


    def build_pointnet(self, X, is_training, bn_decay, scope, reuse=False):
        X_expanded = tf.expand_dims(X, -1)

        net = tf_util.conv2d(X_expanded, 64, [1,3], padding='VALID',
                stride=[1,1], bn=True, is_training=is_training,
                bn_decay=bn_decay, scope=scope+'_conv1', reuse=reuse)

        net = tf_util.conv2d(net, 64, [1,1], padding='VALID',
                stride=[1,1], bn=True, is_training=is_training,
                bn_decay=bn_decay, scope=scope+'_conv2', reuse=reuse)

        net = tf_util.conv2d(net, 64, [1,1], padding='VALID',
                stride=[1,1], bn=True, is_training=is_training,
                bn_decay=bn_decay, scope=scope+'_conv3', reuse=reuse)

        net = tf_util.conv2d(net, 128, [1,1], padding='VALID',
                stride=[1,1], bn=True, is_training=is_training,
                bn_decay=bn_decay, scope=scope+'_conv4', reuse=reuse)

        net = tf_util.conv2d(net, 1024, [1,1], padding='VALID',
                stride=[1,1], bn=True, is_training=is_training,
                bn_decay=bn_decay, scope=scope+'_conv5', reuse=reuse)

        net = tf_util.max_pool2d(net, [self.n_points, 1], padding='VALID',
                scope=scope+'_maxpool', reuse=reuse)

        net = tf.squeeze(net)
        feature = net

        net = tf_util.fully_connected(net, 512, bn=True,
                is_training=is_training, bn_decay=bn_decay,
                scope=scope+'_fc1', reuse=reuse)

        net = tf_util.fully_connected(net, 256, bn=True,
                is_training=is_training, bn_decay=bn_decay,
                scope=scope+'_fc2', reuse=reuse)

        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                scope=scope+'_dp1', reuse=reuse)

        return net, feature


    def build_mixture_density_params(self, net, max_sigma, scope, reuse=False):
        log_logits = tf_util.fully_connected(net, self.K,
                activation_fn=None, scope=scope+'_log_logits', reuse=reuse)
        log_logits = tf.nn.log_softmax(log_logits)

        mus = tf_util.fully_connected(net, self.K * self.Yc_dim,
                activation_fn=None, scope=scope+'_mus', reuse=reuse)
        mus = tf.reshape(mus, [tf.shape(mus)[0], self.K, self.Yc_dim])

        sigmas = tf_util.fully_connected(net, self.K * self.Yc_dim,
                activation_fn=None, scope=scope+'_sigmas', reuse=reuse)
        sigmas = tf.exp(sigmas)

        # Clip sigmas.
        sigmas = tf.clip_by_value(sigmas, 1.0e-6, max_sigma)

        sigmas = tf.reshape(sigmas, [tf.shape(sigmas)[0], self.K, self.Yc_dim])

        return log_logits, mus, sigmas


    def compute_gaussian_mixture_loss(self, Y, log_logits, mus, sigmas):
        # Y: (N x D)
        # log_logits: (N x K)
        # mus, sigmas: (N x K x D)

        Y_expanded = tf.expand_dims(Y, 1)
        # Y_expanded: (N x 1 x D)
        # mus, sigmas: (N x K x D)

        # Pr = 1 / ((2 * pi)^0.5 * sigma) * exp(-0.5 * ((x - mu)/sigma)^2)
        # = exp(-0.5 * ((x - mu)/sigma)^2 - log((2 * pi)^0.5 * sigma))
        # = exp(-0.5 * ((x - mu)/sigma)^2 - 0.5 * log(2 * pi) - log(sigma))
        # = exp(log_probs)

        halfLogTwoPI = 0.5 * math.log(2 * math.pi)
        Y_normalized = tf.multiply(Y_expanded - mus, tf.reciprocal(sigmas))
        log_probs = -0.5 * tf.square(Y_normalized) - halfLogTwoPI -\
                tf.log(sigmas)

        # Assume a diagonal covariance matrix.
        # The probability is simply the 'product' of each dimension
        # probabilities, which is 'sum' in the exponential space.
        # http://cs229.stanford.edu/section/gaussians.pdf
        log_probs = tf.reduce_sum(log_probs, 2)
        # log_probs: (N x K)

        # -log(sum( logit_i * exp(log_probs_i) ))
        # -log(sum( exp(log(logit_i)) * exp(log_probs_i) ))
        # -log(sum( exp(log_probs_i + log(logit_i)) ))
        log_probs = log_probs + log_logits
        # log_probs: (N x K)

        losses = -tf.reduce_logsumexp(log_probs, 1)
        # losses: (N x 1)

        probs = tf.exp(-losses)
        # probs: (N x 1)

        return losses, probs


    def compute_min_dists_to_means(self, Y, mus):
        # Y: (N x D)
        # mus: (N x K x D)

        Y_expanded = tf.expand_dims(Y, 1)
        # Y_expanded: (N x 1 x D)
        # mus, sigmas: (N x K x D)

        sqr_dists = tf.reduce_sum(tf.square(Y_expanded - mus), 2)
        # sqr_dists: (N x K)

        min_sqr_dists = tf.reduce_min(sqr_dists, 1)
        min_dists = tf.sqrt(min_sqr_dists)
        # min_dists: (N x 1)

        return min_dists

