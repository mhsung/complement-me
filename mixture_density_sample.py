# -*- coding: utf-8 -*-
# Minhyuk Sung (mhsung@cs.stanford.edu)
# April 2017

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

from dataset import Dataset
import math
import numpy as np
from scipy import spatial, stats
import tensorflow as tf
import tf_util


def generate_embedding_outputs(sess, net, data, out_dir, loss_func):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    X, Y, _, _, target_idx_list, given_idxs_list, _,\
            possible_target_idxs_list = data.generate_all_X_and_Y()

    n_data = len(X)
    assert (len(target_idx_list) == n_data)
    assert (len(given_idxs_list) == n_data)

    with open(os.path.join(out_dir, 'target_idx_list.csv'), 'w') as f:
        for target_idx in target_idx_list:
            f.write(str(target_idx) + '\n')
    print("Saved 'target_idx_list.csv'.")

    with open(os.path.join(out_dir, 'given_idxs_list.csv'), 'w') as f:
        for given_idxs in given_idxs_list:
            f.write(','.join([str(x) for x in given_idxs]) + '\n')
    print("Saved 'given_idxs_list.csv'.")

    with open(os.path.join(out_dir, 'possible_target_idxs_list.csv'), 'w') as f:
        for possible_target_idxs in possible_target_idxs_list:
            f.write(','.join([str(x) for x in possible_target_idxs]) + '\n')
    print("Saved 'possible_target_idxs_list.csv'.")

    np.save(os.path.join(out_dir, 'given_X.npy'), X)
    print("Saved 'given_X.npy'.")

    # Predict MoG distributions.
    logits, mus, sigmas = predict_MDN_X(sess, net, X)
    np.save(os.path.join(out_dir, 'pred_logits.npy'), logits)
    print("Saved 'pred_logits.npy'.")
    np.save(os.path.join(out_dir, 'pred_embed_mus.npy'), mus)
    print("Saved 'pred_embed_mus.npy'.")
    np.save(os.path.join(out_dir, 'pred_embed_sigmas.npy'), sigmas)
    print("Saved 'pred_embed_sigmas.npy'.")

    # Predict all embedding coordinates.
    pred_Yc = predict_Yc(sess, net, data.centered_points)
    np.save(os.path.join(out_dir, 'pred_Yc.npy'), pred_Yc)
    print("Saved 'pred_Yc.npy'.")


def generate_position_outputs(sess, net, data, out_dir):
    assert(os.path.exists(os.path.join(out_dir, 'given_X.npy')))
    X = np.load(os.path.join(out_dir, 'given_X.npy'))
    print("Loaded 'given_X.npy'.")
    n_data = len(X)

    assert(os.path.exists(os.path.join(out_dir, 'sample_retrieved_idxs.csv')))
    sample_retrieved_idxs = np.loadtxt(os.path.join(
        out_dir, 'sample_retrieved_idxs.csv'), dtype=int, delimiter=',')
    print("Loaded 'sample_retrieved_idxs.csv'.")
    assert(sample_retrieved_idxs.shape[0] == n_data)

    sample_positions = np.empty((sample_retrieved_idxs.shape[0],
        sample_retrieved_idxs.shape[1], 3))

    for i in range(sample_retrieved_idxs.shape[1]):
        Yi = data.centered_points[sample_retrieved_idxs[:, i]]
        assert(Yi.shape[0] == n_data)
        assert(Yi.shape[-1] == 3)
        sample_positions[:, i, :] = predict_Yp(sess, net, X, Yi)

    np.save(os.path.join(out_dir, 'sample_positions.npy'), sample_positions)
    print("Saved 'sample_positions.npy'.")


def predict_MDN_X(sess, net, X):
    n_data = X.shape[0]
    assert (n_data > 0)
    n_batches_in_epoch = int(math.ceil(float(n_data) / net.batch_size))

    logits = None
    mus = None
    sigmas = None

    for index_in_epoch in range(n_batches_in_epoch):
        start = index_in_epoch * net.batch_size
        end = min(start + net.batch_size, n_data)
        n_step_size = end - start
        step_X = X[start:end]

        # NOTE:
        # Add dummy.
        if n_step_size < net.batch_size:
            assert(X.ndim > 1)
            step_X = np.vstack((step_X,
                X[0:(net.batch_size - n_step_size)]))

        step_logits, step_mus, step_sigmas = sess.run(
                [net.logits, net.mus, net.sigmas], feed_dict={
                    net.X: step_X, net.is_training: False})

        # NOTE:
        # Remove dummy data.
        step_logits = step_logits[:n_step_size]
        step_mus = step_mus[:n_step_size]
        step_sigmas = step_sigmas[:n_step_size]

        if index_in_epoch == 0:
            logits = step_logits
            mus = step_mus
            sigmas = step_sigmas
        else:
            logits = np.vstack((logits, step_logits))
            mus = np.vstack((mus, step_mus))
            sigmas = np.vstack((sigmas, step_sigmas))

    # Order by logits.
    for i in range(logits.shape[0]):
        sorted_idxs = np.argsort(logits[i])[::-1]
        logits[i] = logits[i][sorted_idxs]
        mus[i] = mus[i][sorted_idxs]
        sigmas[i] = sigmas[i][sorted_idxs]

    return logits, mus, sigmas


def predict_Yc(sess, net, Y):
    n_data = Y.shape[0]
    assert (n_data > 0)
    n_batches_in_epoch = int(math.ceil(float(n_data) / net.batch_size))

    Yc = None

    for index_in_epoch in range(n_batches_in_epoch):
        start = index_in_epoch * net.batch_size
        end = min(start + net.batch_size, n_data)
        n_step_size = end - start
        step_Y = Y[start:end]

        # NOTE:
        # Add dummy.
        if n_step_size < net.batch_size:
            assert(Y.ndim > 1)
            step_Y = np.vstack((step_Y,
                Y[0:(net.batch_size - n_step_size)]))

        step_Yc = sess.run(net.pred_Yc, feed_dict={
            net.Y: step_Y, net.is_training: False})

        # NOTE:
        # Remove dummy data.
        step_Yc = step_Yc[:n_step_size]

        if index_in_epoch == 0:
            Yc = step_Yc
        else:
            Yc = np.vstack((Yc, step_Yc))

    return Yc


def predict_Yp(sess, net, X, Y):
    n_data = X.shape[0]
    assert (n_data > 0)
    assert (Y.shape[0] == n_data)
    n_batches_in_epoch = int(math.ceil(float(n_data) / net.batch_size))

    Yp = None

    for index_in_epoch in range(n_batches_in_epoch):
        start = index_in_epoch * net.batch_size
        end = min(start + net.batch_size, n_data)
        n_step_size = end - start
        step_X = X[start:end]
        step_Y = Y[start:end]

        # NOTE:
        # Add dummy.
        if n_step_size < net.batch_size:
            assert(X.ndim > 1)
            step_X = np.vstack((step_X,
                X[0:(net.batch_size - n_step_size)]))
            step_Y = np.vstack((step_Y,
                Y[0:(net.batch_size - n_step_size)]))

        step_Yp = sess.run(net.pred_Yp, feed_dict={
            net.X: step_X, net.Y: step_Y, net.is_training: False})

        # NOTE:
        # Remove dummy data.
        step_Yp = step_Yp[:n_step_size]

        if index_in_epoch == 0:
            Yp = step_Yp
        else:
            Yp = np.vstack((Yp, step_Yp))

    return Yp


def sample_logit_idx(x, pdf):
    N = pdf.size
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
            return i

    raise AssertionError


def generate_ensembles(logits, mus, sigmas, n_samples):
    # logits: (N x K)
    # mus, sigmas: (N x K x D)

    n_data = mus.shape[0]
    dim = mus.shape[-1]
    rn = np.random.rand(n_data)

    # samples: (N x D)
    samples = np.empty((n_data, n_samples, dim))
    probs = np.empty((n_data, n_samples))

    # transforms samples into random ensembles
    for i in range(0, n_data):
        for j in range(0, n_samples):
            idx = sample_logit_idx(rn[i], logits[i])
            mean = mus[i, idx]
            std = sigmas[i, idx]
            assert(std.shape[0] > 1)
            #samples[i, j] = np.random.normal(mean, std)
            samples[i, j] = np.random.multivariate_normal(mean, np.diag(std))
            probs[i, j] = logits[i, idx] * stats.multivariate_normal(
                    mean=mean, cov=np.diag(std)).pdf(samples[i, j])

        # Order by probabilities.
        sorted_idxs = np.argsort(probs[i])[::-1]
        probs[i] = probs[i][sorted_idxs]
        samples[i] = samples[i][sorted_idxs]

    return samples, probs


def retrieve_nearest_neighbors(data, embed_coords):
    n_data = embed_coords.shape[0]
    k = embed_coords.shape[1]

    # NOTE:
    # Use 'all' components as a pool of retrieval.
    tree = spatial.KDTree(data.normalized_embedding_coords)
    retrieved_idxs = np.empty((n_data, k), dtype=int)
    for i in range(n_data):
        for j in range(k):
            _, retrieved_idxs[i,j] = tree.query(embed_coords[i,j], k=1)

    return retrieved_idxs

