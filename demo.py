
from pathlib import Path
import argparse
import random

import cv2
import numpy as np
import matplotlib.cm as cm
import torch
import slam_lib
import slam_lib.feature
# from models.matching import Matching
# from models.utils import (compute_pose_error, compute_epipolar_error,
#                           estimate_pose, make_matching_plot,
#                           error_colormap, AverageTimer, pose_auc, read_image,
#                           rotate_intrinsics, rotate_pose_inplane,
#                           scale_intrinsics)
import slam_lib.vis


torch.set_grad_enabled(False)
device = 'cuda:0'
resize = np.asarray([3088, 2064]) // 2

config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }

# img_1 = './LeftCamera-Snapshot-20220221170353-29490143290143.BMP'
# img_2 = './RightCamera-Snapshot-20220221170350-29490115460981.BMP'


img_1 = '/home/cheng/proj/3d/SuperGluePretrainedNetwork/LeftCamera-Snapshot-20220221170353-29490143290143_small.BMP'
img_2 = '/home/cheng/proj/3d/SuperGluePretrainedNetwork/RightCamera-Snapshot-20220221170350-29490115460981_small.BMP'


matcher = slam_lib.feature.Matcher()

# gray_0, inp0, scales0 = read_image(img_1, device, resize, rotation=0, resize_float=True)
# gray_1, inp1, scales1 = read_image(img_2, device, resize, rotation=0, resize_float=True)
#
# pred = matching({'image0': inp0, 'image1': inp1})
#
#
# pts_2d_0, pts_2d_1 = slam_lib.mapping.scale_pts(scales0, mkpts_0), slam_lib.mapping.scale_pts(scales1, mkpts_1)
#
# if flag_vis:
#     img3 = slam_lib.vis.draw_matches(img_left, pts_2d_0, img_right, pts_2d_1)
#     cv2.namedWindow('match', cv2.WINDOW_NORMAL)
#     cv2.imshow('match', img3)
#     cv2.waitKey(0)

pts_2d_left_super, _, pts_2d_right_super, _ = matcher.match(img_1, img_2)

img1 = cv2.imread(img_1)
img2 = cv2.imread(img_2)
img = slam_lib.vis.draw_matches(img1, pts_2d_left_super, img2, pts_2d_right_super)

cv2.namedWindow('match', cv2.WINDOW_NORMAL)
cv2.imshow('match', img)
cv2.waitKey(0)
