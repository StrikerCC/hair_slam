# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 9/2/21 5:46 PM
"""
import copy
import time

import cv2
import numpy as np
from scipy.spatial.kdtree import KDTree

import read
import utils

def compute_fpf(img, start_points, end_points, num_neighbor):
    """"""
    # num_neighbor = 5
    num_hair = len(start_points)

    assert start_points.shape[1] == 2 and end_points.shape[1] == 2
    '''setup local frame for each point'''
    # use hair line as first axis, counter-clockwise as positive rotation
    rotation_matrix_90_counter_clockwise = np.array([[0, 1],
                                                     [1, -1]])
    hair_line = end_points - start_points
    hair_line_unit_vector = hair_line / np.expand_dims(np.linalg.norm(hair_line, axis=1), axis=-1)
    hair_norm_unit_vector = np.dot(rotation_matrix_90_counter_clockwise, hair_line_unit_vector.T).T
    assert 0.999 < np.linalg.norm(hair_line_unit_vector[0]) < 1.0001, np.linalg.norm(hair_line_unit_vector[0])

    '''build tree for hair data, key is hair start coordinate, value is hair orientation'''
    tree_start = KDTree(start_points)

    '''use orientation and distance of knn feature to make extra dimension data, start from neighbor close to first axis'''
    feature_dis = np.zeros((num_hair, num_neighbor))
    feature_orientation_neighbor_line = np.zeros((num_hair, num_neighbor))
    feature_orientation_neighbor_hair = np.zeros((num_hair, num_neighbor))

    for i_hair in range(num_hair):
        hair_point = start_points[i_hair]
        _, nn_index_list = tree_start.query(hair_point, k=num_neighbor+1)
        nn_index_list = nn_index_list[1:]   # get rid of point itself

        for i_nn, index_nn in enumerate(nn_index_list):
            feature_dis[i_hair, i_nn] = np.linalg.norm(start_points[i_hair] - start_points[index_nn])   # distance
            feature_orientation_neighbor_line[i_hair, i_nn] = np.dot(start_points[index_nn]-start_points[i_hair].T, hair_norm_unit_vector[i_hair]) / feature_dis[i_hair, i_nn]
            feature_orientation_neighbor_line[i_hair, i_nn] /= np.dot(start_points[index_nn]-start_points[i_hair].T, hair_line_unit_vector[i_hair]) / feature_dis[i_hair, i_nn]      # orientation of line segment from hair start to neighbor hair start in local frame
            feature_orientation_neighbor_hair[i_hair, i_nn] = np.dot(hair_line_unit_vector[index_nn].T,
                                                                     hair_line_unit_vector[i_hair])  # orientation of hair in local frame

        # reorganize the nn list, start from point heading close to first axis
        mask_sorted = np.argsort(feature_orientation_neighbor_line[i_hair])
        feature_dis[i_hair] = feature_dis[i_hair][mask_sorted]
        feature_orientation_neighbor_line[i_hair] = feature_orientation_neighbor_line[i_hair][mask_sorted]
        feature_orientation_neighbor_hair[i_hair] = feature_orientation_neighbor_hair[i_hair][mask_sorted]

    feature_dis /= np.expand_dims(np.sum(feature_dis, axis=1), axis=-1)      # normalize distance
    feature_orientation_neighbor_line = np.arctan(feature_orientation_neighbor_line)    # orientation of line segment from hair start to neighbor hair start in local frame


    '''build new feature tree from new feature'''
    # feature = np.concatenate([feature_dis, feature_orientation_neighbor_line, feature_orientation_neighbor_hair], axis=1)
    feature = np.concatenate([feature_dis, feature_orientation_neighbor_line], axis=1)
    return feature


def main():
    time_0 = time.time()
    vis_flag = False
    num_nn = 5
    img_l, data = read.format_data()
    start_points_l, end_points_l = read.dic_2_nparray(img_l, data)
    img_r, start_points_r, end_points_r = utils.rotate(img_l, start_points_l, end_points_l)

    feature_1 = compute_fpf(img_l, start_points_l, end_points_l, num_neighbor=num_nn)
    feature_2 = compute_fpf(img_r, start_points_r, end_points_r, num_neighbor=num_nn)

    '''for each hair, locate it nn in new feature space'''
    num_hair = len(feature_1)
    num_success = 0
    tree_feature_2 = KDTree(feature_2)
    num_neighbor_feature = 1

    for i_hair in range(num_hair):
        img_feature = np.concatenate([img_l, img_r], axis=1)
        hair_feature = feature_1[i_hair]
        _, nn_index_list = tree_feature_2.query(hair_feature, k=num_neighbor_feature + 1)
        if nn_index_list[0] == i_hair:
            num_success += 1
        if vis_flag:
            print(i_hair)
            # for index_nn in nn_index_list:
            index_nn = nn_index_list[0]
            print('     ', index_nn)
            start_point_r = start_points_r[index_nn].astype(int)
            start_point_r += np.asarray([img_l.shape[1], 0])
            cv2.line(img_feature, start_points_l[i_hair], start_point_r, color=(0, 0, 155), thickness=2)
            cv2.imshow('line', img_feature)
            cv2.waitKey(0)
            # cv2.destroyAllWindows()

    '''out'''
    print('success rate', num_success/num_hair, 'time', time.time() - time_0)


if __name__ == '__main__':
    main()
