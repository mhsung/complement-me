# -*- coding: utf-8 -*-
# Minhyuk Sung (mhsung@cs.stanford.edu)
# April 2017

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

from global_variables import *
from resample_points import resample_points
import math
import numpy as np
import pandas as pd
import random


class Dataset(object):
    def __init__(self, pairs_file, batch_size, D):
        self.batch_size = batch_size
        self.Yc_dim = D

        assert(os.path.exists(g_component_all_component_labels_list))
        df = pd.read_csv(g_component_all_component_labels_list,
                header=0, index_col=None)
        self.all_md5s = df['md5'].unique().tolist()
        self.all_comp_ids = df['idx'].unique().tolist()

        assert(os.path.exists(g_component_all_centered_point_clouds_file))
        self.centered_point_clouds = np.load(
                g_component_all_centered_point_clouds_file)
        self.n_components = self.centered_point_clouds.shape[0]
        self.n_points = self.centered_point_clouds.shape[1]
        assert(self.centered_point_clouds.shape[2] == 3)

        assert(os.path.exists(g_component_all_positions_file))
        self.positions = np.load(g_component_all_positions_file)
        assert(self.positions.shape[0] == self.n_components)
        assert(self.positions.shape[1] == 4)
        # Split to positions and areas.
        self.areas = self.positions[:, 3]
        self.positions = self.positions[:, :3]

        # Compute point sets in the original positions.
        self.orig_points = self.centered_point_clouds +\
                np.expand_dims(self.positions, axis=1)

        # Read pair list.
        df = pd.read_csv(pairs_file, index_col=False)
        unique_md5_list = df['md5'].unique().tolist()
        self.comp_graphs = []

        for md5 in unique_md5_list:
            pairs = df[df['md5'] == md5]
            d = dict()
            for _, row in pairs.iterrows():
                idx1 = row['idx1']
                idx2 = row['idx2']
                d[idx1] = [idx2] if idx1 not in d else d[idx1] + [idx2]
                d[idx2] = [idx1] if idx2 not in d else d[idx2] + [idx1]

            if not self.is_all_connected(d):
                continue
            self.comp_graphs.append(d)

        self.n_data = len(self.comp_graphs)


    def is_all_connected(self, d):
        assert(d)

        seed_idx = d.keys()[0]
        given_idxs = [seed_idx]
        connected_idxs = []
        while True:
            connected_idxs += d[seed_idx]
            connected_idxs = [x for x in connected_idxs if x not in given_idxs]
            if not connected_idxs:
                break
            connected_idxs = list(set(connected_idxs))
            seed_idx = random.choice(connected_idxs)
            given_idxs.append(seed_idx)

        return True if len(given_idxs) == len(d) else False


    def get_random_subgraph_and_connected_target(self, d):
        assert(d)
        n_given = random.randint(1, len(d) - 1)
        target_idx = random.choice(d.keys())

        seed_idx = target_idx
        given_idxs = []
        connected_idxs = []
        for count in range(n_given):
            connected_idxs += d[seed_idx]
            connected_idxs = [x for x in connected_idxs\
                    if x != target_idx and x not in given_idxs]
            assert(connected_idxs)
            connected_idxs = list(set(connected_idxs))
            seed_idx = random.choice(connected_idxs)
            given_idxs.append(seed_idx)

        connected_idxs = []
        for x in given_idxs:
            connected_idxs += d[x]
        connected_idxs = list(set(connected_idxs))
        possible_target_idxs = [x for x in connected_idxs \
                if x not in given_idxs]
        assert(target_idx in possible_target_idxs)

        return target_idx, given_idxs, possible_target_idxs


    def generate_random_X_and_Y(self, graph_idxs):
        n_graphs = graph_idxs.size

        X = np.empty((n_graphs, self.centered_point_clouds.shape[1],
            self.centered_point_clouds.shape[2]))

        # Positive example.
        Y = np.empty((n_graphs, self.centered_point_clouds.shape[1],
            self.centered_point_clouds.shape[2]))
        Yc = np.empty((n_graphs, self.Yc_dim))
        Yp = np.empty((n_graphs, 3))

        # Negative example.
        Z = np.empty((n_graphs, self.centered_point_clouds.shape[1],
            self.centered_point_clouds.shape[2]))
        Zc = np.empty((n_graphs, self.Yc_dim))

        target_idx_list = []
        given_idxs_list = []
        neg_sample_idx_list = []
        possible_target_idxs_list = []

        for i in range(n_graphs):
            graph_idx = graph_idxs[i]
            d = self.comp_graphs[graph_idx]

            target_idx, given_idxs, possible_target_idxs =\
                    self.get_random_subgraph_and_connected_target(d)

            # Resample points in the all other components.
            given_points_resampled = resample_points(
                    self.orig_points[given_idxs],
                    self.areas[given_idxs])

            # Take a negative example.
            neg_sample_idx = random.choice(
                    [x for x in range(self.n_components) if x != target_idx])

            X[i] = given_points_resampled
            Y[i] = self.centered_point_clouds[target_idx]
            Yp[i] = self.positions[target_idx]
            Z[i] = self.centered_point_clouds[neg_sample_idx]

            target_idx_list.append(target_idx)
            given_idxs_list.append(given_idxs)
            neg_sample_idx_list.append(neg_sample_idx)
            possible_target_idxs_list.append(possible_target_idxs)

        return X, Y, Yp, Z, target_idx_list, given_idxs_list,\
                neg_sample_idx_list, possible_target_idxs_list


    def generate_all_X_and_Y(self):
        # Use all models if the graph indices are not given.
        return self.generate_random_X_and_Y(np.arange(self.n_data))


    def __iter__(self):
        self.index_in_epoch = 0
        self.perm = np.arange(self.n_data)
        np.random.shuffle(self.perm)
        return self


    def next(self):
        self.start = self.index_in_epoch * self.batch_size

        # FIXME:
        # Fix this when input placeholders have dynamic sizes.
        #self.end = min(self.start + self.batch_size, self.n_data)
        self.end = self.start + self.batch_size

        self.step_size = self.end - self.start
        self.index_in_epoch = self.index_in_epoch + 1

        # FIXME:
        # Fix this when input placeholders have dynamic sizes.
        #if self.start < self.n_data:
        if self.end <= self.n_data:
            shuffled_indices = self.perm[self.start:self.end]
            step_X, step_Y, step_Yp, step_Z _, _, _, _ =\
                    self.generate_random_X_and_Y(shuffled_indices)
            return step_X, step_Y, step_Yp, step_Z
        else:
            raise StopIteration()

