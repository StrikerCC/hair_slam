# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/31/21 4:43 PM
"""
import time

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
    square_size = 3.0
    checkboard_size = (11, 8)  # (board_width, board_height)

    dataset_dir = '/home/cheng/Pictures/data/202201111639'
    data = dataset.get_calibration_and_img(dataset_dir)
    # binocular = BiCamera('./config/bicam_cal_para.json')
    binocular = BiCamera()
    stereo_calibrate(square_size, checkboard_size, data['left_calibration_img'], data['right_calibration_img'], binocular=binocular,
                     file_path_2_save='./config/bicam_cal_para.json')

    print('calibration result')
    print(binocular.cam_left.camera_matrix)
    print(binocular.cam_right.camera_matrix)

    # 3d recon
    for i, (img_left_path, img_right_path) in enumerate(zip(data['left_general_img'], data['right_general_img'])):
        print('running checkboard rec', i, img_left_path, img_right_path)
        # preprocess image
        img_left, img_right = cv2.imread(img_left_path), cv2.imread(img_right_path)
        img_left, img_right = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        img_left_rectified, img_right_rectified = binocular.cam_left.undistort_rectify_img(
            img_left), binocular.cam_right.undistort_rectify_img(img_right)

        # extract feature
        time_start = time.time()
        pts_2d_left = feature.get_checkboard_corners(img_left, checkboard_size)
        pts_2d_right = feature.get_checkboard_corners(img_right, checkboard_size)

        pts_2d_left_rectified = feature.get_checkboard_corners(img_left_rectified, checkboard_size)
        pts_2d_right_rectified = feature.get_checkboard_corners(img_right_rectified, checkboard_size)

        print('get feature in', time.time() - time_start, 'seconds')
        if pts_2d_left_rectified is None or pts_2d_right_rectified is None:
            if pts_2d_left_rectified is None: print('feature missed in left rectification')
            if pts_2d_right_rectified is None: print('feature missed in right rectification')
            continue
        # compute feature 3d coord
        pts_3d_from_raw = binocular.transform_raw_pixel_to_world_coordiante(pts_2d_left, pts_2d_right)
        pts_3d_from_rect = binocular.transform_rectify_pixel_to_world_coordiante(pts_2d_left_rectified,
                                                                                 pts_2d_right_rectified)

        # statistics
        dist = pts_3d_from_raw[:-1, :] - pts_3d_from_raw[1:, :]
        dist = np.linalg.norm(dist, axis=1)
        dist = dist[dist < square_size*2]
        print('average distance between checkbaord corners', dist.mean())

        # vis
        mesh = o3.geometry.TriangleMesh()
        frame = mesh.create_coordinate_frame(size=25.0)
        pc_raw = o3.geometry.PointCloud()
        pc_raw.points = o3.utility.Vector3dVector(pts_3d_from_raw)

        pc_rect = o3.geometry.PointCloud()
        pc_rect.points = o3.utility.Vector3dVector(pts_3d_from_rect)

        o3.visualization.draw_geometries([pc_raw, pc_rect, frame])
        o3.visualization.draw_geometries([pc_raw, pc_rect])


if __name__ == '__main__':
    main()
