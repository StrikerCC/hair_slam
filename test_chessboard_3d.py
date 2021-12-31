# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/31/21 4:43 PM
"""
import cv2
import numpy as np
import open3d as o3

import dataset
import feature
from camera.cam import BiCamera
from camera.calibration import stereo_calibrate


def main():
    """calibration and 3d reconstruction"""

    # cal
    square_size = 0.02423
    checkboard_size = (6, 9)  # (board_width, board_height)

    img_dir = './data/1/'
    img_left_paths, img_right_paths = dataset.get_left_right_img_path_in_one_folder(img_dir)
    binocular = BiCamera()
    stereo_calibrate(square_size, checkboard_size, img_left_paths, img_right_paths, binocular=binocular,
                     file_2_save='./config/bicam_cal_para.json')

    print('calibration result')
    print(binocular.cam_left.camera_matrix)
    print(binocular.cam_right.camera_matrix)

    '''TODO: get Q matrix from calibration'''

    # 3d recon
    for i, (img_left_path, img_right_path) in enumerate(zip(img_left_paths, img_right_paths)):
        # preprocess image
        img_left, img_right = cv2.imread(img_left_path), cv2.imread(img_right_path)
        img_left, img_right = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # extract feature
        pts_2d_left = feature.get_checkboard_corners(img_left, checkboard_size)
        pts_2d_right = feature.get_checkboard_corners(img_right, checkboard_size)
        pts_3d = binocular.transform_pixel_to_world_coordiante(pts_2d_left, pts_2d_right)

        # vis
        pc = o3.geometry.PointCloud()
        pc.points = o3.utility.Vector3dVector(pts_3d)
        o3.visualization.draw_geometries([pc])


if __name__ == '__main__':
    main()
