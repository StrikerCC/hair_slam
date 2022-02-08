# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/10/21 3:58 PM
"""
import cv2
import slam_lib.feature


def find_homo(img_src, img_tgt, flag_vis_feature_matching=False):
    # pts_src, des_src, pts_tgt, des_tgt = slam_lib.feature.get_sift_and_pts(img_src, img_tgt, flag_debug=flag_vis_feature_matching)
    pts_src, des_src, pts_tgt, des_tgt = slam_lib.feature.match_pts(img_src, img_tgt, flag_vis_feature_matching)
    if pts_src.shape[0] < 0 or pts_tgt.shape[0] < 4:
        print('Not enough points ' + str(pts_src.shape[0]) + ' and ' + str(pts_tgt.shape[0]))
        return None
    homo, status = cv2.findHomography(pts_src, pts_tgt, method=cv2.RANSAC, ransacReprojThreshold=4)
    return homo
