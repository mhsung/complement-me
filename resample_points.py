#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Minhyuk Sung (mhsung@cs.stanford.edu)
# January 2017

import numpy as np
import random


class Walkerrandom:
    """ Walker's alias method for random objects with different probablities
    http://code.activestate.com/recipes/576564-walkers-alias-method-for-random-objects-with-diffe/
    """

    def __init__(self, weights, keys=None):
        """ builds the Walker tables prob and inx for calls to random().
        The weights (a list or tuple or iterable) can be in any order;
        they need not sum to 1.
        """

        n = self.n = len(weights)
        self.keys = keys
        sumw = sum(weights)
        prob = [w * n / sumw for w in weights]  # av 1
        inx = [-1] * n
        short = [j for j, p in enumerate(prob) if p < 1]
        long = [j for j, p in enumerate(prob) if p > 1]
        while short and long:
            j = short.pop()
            k = long[-1]
            # assert prob[j] <= 1 <= prob[k]
            inx[j] = k
            prob[k] -= (1 - prob[j])  # -= residual weight
            if prob[k] < 1:
                short.append(k)
                long.pop()
            #if Test:
            #    print "test Walkerrandom: j k pk: %d %d %.2g" % (j, k, prob[k])
        self.prob = prob
        self.inx = inx
        #if Test:
        #    print "test", self


    def __str__(self):
        """ e.g. "Walkerrandom prob: 0.4 0.8 1 0.8  inx: 3 3 -1 2" """

        probstr = " ".join(["%.2g" % x for x in self.prob])
        inxstr = " ".join(["%.2g" % x for x in self.inx])
        return "Walkerrandom prob: %s  inx: %s" % (probstr, inxstr)


    def random(self):
        """ each call -> a random int or key with the given probability
        fast: 1 randint(), 1 random.uniform(), table lookup
        """

        u = random.uniform(0, 1)
        j = random.randint(0, self.n - 1)  # or low bits of u
        randint = j if u <= self.prob[j] else self.inx[j]
        return self.keys[randint] if self.keys else randint


def resample_points(points, scales, n_out_points=None):
    assert (points.ndim == 3)
    assert (scales.ndim == 1)
    assert (points.shape[0] == scales.size)

    n_sets = points.shape[0]
    n_in_points = points.shape[1]
    n_dim = points.shape[2]
    if n_out_points is None:
        n_out_points = points.shape[1]

    wrand = Walkerrandom(scales)
    set_idxs = np.empty(n_out_points, dtype='int')
    for i in range(n_out_points):
        set_idxs[i] = wrand.random()

    point_rand_idxs_in_sets = np.random.randint(n_in_points, size=n_out_points)

    sample_points = points[set_idxs, point_rand_idxs_in_sets, :]
    return sample_points


def axis_aligned_bbox_center(points):
    assert (points.ndim == 2)
    bb_min = np.nanmin(points, axis=0)
    bb_max = np.nanmax(points, axis=0)
    bb_center = 0.5 * (bb_min + bb_max)
    return bb_center


def centerize_points(points):
    assert (points.ndim == 2)
    bb_center = axis_aligned_bbox_center(points)
    centered_points = points - np.expand_dims(bb_center, 0)
    return centered_points, bb_center

