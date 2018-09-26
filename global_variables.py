#!/usr/bin/python
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# Define all the global variables for the project
#------------------------------------------------------------------------------

from __future__ import division
import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

#g_shape_synset = 'Chair'
g_shape_synset = os.environ['synset']


### SET DATA ROOT DIRECTORY ###
g_component_root_dir = os.path.join(BASE_DIR, 'data', 'components')
g_component_synset_dir = os.path.join(g_component_root_dir, g_shape_synset)


g_component_all_component_labels_list = os.path.abspath(os.path.join(
    g_component_synset_dir, 'component_all_component_labels.txt'))

g_component_all_centered_point_clouds_file = os.path.abspath(os.path.join(
    g_component_synset_dir, 'component_all_centered_point_clouds.npy'))

g_component_all_positions_file = os.path.abspath(os.path.join(
    g_component_synset_dir, 'component_all_positions.npy'))


g_component_all_md5_list = os.path.abspath(os.path.join(
    g_component_synset_dir, 'component_all_md5s.txt'))

g_component_train_md5_list = os.path.abspath(os.path.join(
    g_component_synset_dir, 'component_train_md5s.txt'))

g_component_test_md5_list = os.path.abspath(os.path.join(
    g_component_synset_dir, 'component_test_md5s.txt'))


g_component_all_pairs_file = os.path.abspath(os.path.join(
    g_component_synset_dir, 'component_all_pairs.csv'))

g_component_train_pairs_file = os.path.abspath(os.path.join(
    g_component_synset_dir, 'component_train_pairs.csv'))

g_component_test_pairs_file = os.path.abspath(os.path.join(
    g_component_synset_dir, 'component_test_pairs.csv'))
