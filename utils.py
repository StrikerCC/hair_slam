# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 9/7/21 3:57 PM
"""
import copy
import math

import cv2
import numpy as np


def rot2d(angle):
    return np.asarray([[math.cos(angle), -math.sin(angle)],
                       [math.sin(angle), math.cos(angle)]])


def affine2d(angle, x=0, y=0):
    rot = rot2d(angle)
    translation = np.asarray([[x],[y]])
    return np.concatenate([rot, translation], axis=1)

    
def rotate(img, start_points, end_points, angle=math.pi/224, flag_vis=False):
    img_r, start_points_r, end_points_r = copy.deepcopy((img, start_points, end_points))
    start_points_r, end_points_r = np.concatenate([start_points_r, np.ones((len(start_points_r), 1))], axis=1), \
                                   np.concatenate([end_points_r, np.ones((len(end_points_r), 1))], axis=1)
    affine = affine2d(angle, x=50, y=20)
    img_r = cv2.warpAffine(img_r, M=affine, dsize=(img_r.shape[1], img_r.shape[0]))
    start_points_r = np.dot(affine, start_points_r.T).T
    end_points_r = np.dot(affine, end_points_r.T).T
    if flag_vis:
        for start_point_r in start_points_r:
            if np.alltrue(start_point_r > 0) and np.alltrue(start_point_r < np.asarray(img_r.shape[:2])[::-1]):
                start_point_r = start_point_r.astype(int)
                print(start_point_r)
                cv2.circle(img_r, start_point_r, 10, color=(0, 0, 255), thickness=3, )
                cv2.imshow('hairs rotated', img_r)
                cv2.waitKey(0)

    return img_r, start_points_r, end_points_r
