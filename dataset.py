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
    def __init__(self, pairs_file, batch_size, D,
            single_target=False, single_given=False):
        self.batch_size = batch_size
        self.Yc_dim = D
        self.single_target = single_target
        self.single_given = single_given

        if single_target:
            assert(not single_given)
            print("\n!! RUNNING WITH SINGLE TARGETS !!\n")
        elif single_given:
            assert(not single_target)
            print("\n!! RUNNING WITH SINGLE GIVEN COMPONENTS !!\n")

        assert(os.path.exists(g_group_all_centered_points_list))
        with open(g_group_all_centered_points_list) as f:
            self.centered_points_files = f.read().splitlines()

        self.all_md5s = []
        self.all_comp_ids = []
        for i in range(len(self.centered_points_files)):
            md5 = os.path.basename(os.path.dirname(
                self.centered_points_files[i]))
            comp_id = int(os.path.splitext(os.path.basename(
                self.centered_points_files[i]))[0])
            self.all_md5s.append(md5)
            self.all_comp_ids.append(comp_id)

        assert(os.path.exists(g_group_all_centered_points_file))
        self.centered_points = np.load(g_group_all_centered_points_file)
        self.n_components = self.centered_points.shape[0]
        self.n_points = self.centered_points.shape[1]
        assert(self.centered_points.shape[2] == 3)

        assert(os.path.exists(g_group_all_positions_file))
        self.positions = np.load(g_group_all_positions_file)
        assert(self.positions.shape[0] == self.n_components)
        assert(self.positions.shape[1] == 4)
        # Split to positions and areas.
        self.areas = self.positions[:, 3]
        self.positions = self.positions[:, :3]

        # Compute point sets in the original positions.
        self.orig_points = self.centered_points +\
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
        possible_target_idxs = [x for x in connected_idxs if x not in given_idxs]
        assert(target_idx in possible_target_idxs)

        return target_idx, given_idxs, possible_target_idxs


    def generate_random_X_and_Y(self, graph_idxs):
        n_graphs = graph_idxs.size

        X = np.empty((n_graphs,
            self.centered_points.shape[1], self.centered_points.shape[2]))

        # Positive example.
        Y = np.empty((n_graphs,
            self.centered_points.shape[1], self.centered_points.shape[2]))
        Yc = np.empty((n_graphs, self.Yc_dim))
        Yp = np.empty((n_graphs, 3))

        # Negative example.
        Z = np.empty((n_graphs,
            self.centered_points.shape[1], self.centered_points.shape[2]))
        Zc = np.empty((n_graphs, self.Yc_dim))

        target_idx_list = []
        given_idxs_list = []
        neg_sample_idx_list = []
        possible_target_idxs_list = []

        for i in range(n_graphs):
            graph_idx = graph_idxs[i]
            d = self.comp_graphs[graph_idx]

            if self.single_target:
                target_idx = random.choice(d.keys())
                given_idxs = [x for x in d.keys() if x != target_idx]
                possible_target_idxs = [target_idx]
            elif self.single_given:
                given_idxs = [random.choice(d.keys())]
                possible_target_idxs = [x for x in d.keys() \
                        if x != given_idxs[0]]
                target_idx = random.choice(possible_target_idxs)
            else:
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
            Y[i] = self.centered_points[target_idx]
            Yp[i] = self.positions[target_idx]
            Z[i] = self.centered_points[neg_sample_idx]

            target_idx_list.append(target_idx)
            given_idxs_list.append(given_idxs)
            neg_sample_idx_list.append(neg_sample_idx)
            possible_target_idxs_list.append(possible_target_idxs)

        return X, Y, Yp, Z, target_idx_list, given_idxs_list,\
                neg_sample_idx_list, possible_target_idxs_list


    def generate_all_X_and_Y(self):
        if not self.single_target and not self.single_given:
            # Use all models if the graph indices are not given.
            return self.generate_random_X_and_Y(np.arange(self.n_data))

        # Collect all components in the train/test set.
        all_comp_idxs = []
        for i in range(self.n_data):
            d = self.comp_graphs[i]
            all_comp_idxs += d.keys()
        n_all_cases = len(all_comp_idxs)

        X = np.empty((n_all_cases,
            self.centered_points.shape[1], self.centered_points.shape[2]))

        # Positive example.
        Y = np.empty((n_all_cases,
            self.centered_points.shape[1], self.centered_points.shape[2]))
        Yc = np.empty((n_all_cases, self.Yc_dim))
        Yp = np.empty((n_all_cases, 3))

        # Negative example.
        Z = np.empty((n_all_cases,
            self.centered_points.shape[1], self.centered_points.shape[2]))
        Zc = np.empty((n_all_cases, self.Yc_dim))

        target_idx_list = []
        given_idxs_list = []
        possible_target_idxs_list = []
        neg_sample_idx_list = []

        for i in range(n_all_cases):
            if self.single_target:
                target_idx = all_comp_idxs[i]
                given_idxs = [x for x in range(self.n_components) \
                        if self.all_md5s[x] == self.all_md5s[all_comp_idxs[i]] \
                        and x != all_comp_idxs[i]]
                possible_target_idxs = [target_idx]
            elif self.single_given:
                given_idxs = [all_comp_idxs[i]]
                possible_target_idxs = [x for x in range(self.n_components) \
                        if self.all_md5s[x] == self.all_md5s[all_comp_idxs[i]] \
                        and x != all_comp_idxs[i]]
                target_idx = random.choice(possible_target_idxs)
            else:
                raise AssertionError

            # Resample points in the all other components.
            given_points_resampled = resample_points(
                    self.orig_points[given_idxs],
                    self.areas[given_idxs])

            # Take a negative example.
            neg_sample_idx = random.choice(
                    [x for x in range(self.n_components) if x != target_idx])

            X[i] = given_points_resampled
            Y[i] = self.centered_points[target_idx]
            Yp[i] = self.positions[target_idx]
            Z[i] = self.centered_points[neg_sample_idx]

            target_idx_list.append(target_idx)
            given_idxs_list.append(given_idxs)
            neg_sample_idx_list.append(neg_sample_idx)
            possible_target_idxs_list.append(possible_target_idxs)

        return X, Y, Yp, Z, target_idx_list, given_idxs_list,\
                neg_sample_idx_list, possible_target_idxs_list


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

