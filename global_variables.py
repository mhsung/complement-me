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


# Root directories.
g_data_root_dir = ''
g_synset_data_dir = os.path.join(g_data_root_dir, g_shape_synset)


# Executales
g_renderer = '/orions3-zfs/projects/minhyuk/app/libigl-renderer/build/OSMesaRenderer'
g_mesh_merger = '/orions3-zfs/projects/minhyuk/app/libigl-mesh-merger/build/OSMesaRenderer'


# Group directories.
g_group_mesh_root_dir = os.path.abspath(os.path.join(g_synset_data_dir,\
        'group', 'group_mesh'))
g_group_centered_mesh_root_dir = os.path.abspath(os.path.join(g_synset_data_dir,\
        'group', 'group_centered_mesh'))

g_group_points_root_dir = os.path.abspath(os.path.join(g_synset_data_dir,\
        'group', 'group_points_1024'))
g_group_centered_points_root_dir = os.path.abspath(os.path.join(g_synset_data_dir,\
        'group', 'group_centered_points_1024'))
g_group_position_root_dir = os.path.abspath(os.path.join(g_synset_data_dir,\
        'group', 'group_position'))


# Neural net input files.
g_group_all_md5_list = os.path.abspath(os.path.join(g_synset_data_dir,\
        'group', 'neural_net', 'all_md5s.txt'))
g_group_train_md5_list = os.path.abspath(os.path.join(g_synset_data_dir,\
        'group', 'neural_net', 'train_md5s.txt'))
g_group_test_md5_list = os.path.abspath(os.path.join(g_synset_data_dir,\
        'group', 'neural_net', 'test_md5s.txt'))

g_group_all_pairs_file = os.path.abspath(os.path.join(g_synset_data_dir,\
        'group', 'neural_net', 'all_pairs.csv'))
g_group_train_pairs_file = os.path.abspath(os.path.join(g_synset_data_dir,\
        'group', 'neural_net', 'train_pairs.csv'))
g_group_test_pairs_file = os.path.abspath(os.path.join(g_synset_data_dir,\
        'group', 'neural_net', 'test_pairs.csv'))

g_group_all_centered_meshes_list = os.path.abspath(os.path.join(g_synset_data_dir,\
        'group', 'neural_net', 'all_centered_meshes_files.txt'))
g_group_all_centered_points_list = os.path.abspath(os.path.join(g_synset_data_dir,\
        'group', 'neural_net', 'all_centered_points_files.txt'))
g_group_all_centered_points_file = os.path.abspath(os.path.join(g_synset_data_dir,\
        'group', 'neural_net', 'all_centered_points.npy'))
g_group_all_positions_file = os.path.abspath(os.path.join(g_synset_data_dir,\
        'group', 'neural_net', 'all_positions.npy'))

