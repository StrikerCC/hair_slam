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

    # calibration
    square_size = 3.0
    checkboard_size = (11, 8)  # (board_width, board_height)
    flag_reconstruct_checkboard = False

    dataset_dir = '/home/cheng/Pictures/data/202201111639/'
    data = dataset.get_calibration_and_img(dataset_dir)

    '''shorten calibration data'''
    data['left_calibration_img'], data['right_calibration_img'] = data['left_calibration_img'][:5], \
                                                                  data['right_calibration_img'][:5]

    for key in data.keys():
        print(key, 'has')
        for img_dir in data.get(key):
            print('     ', img_dir)

    binocular = BiCamera('./config/bicam_cal_para.json')
    # binocular = BiCamera()
    # stereo_calibrate(square_size, checkboard_size, data['left_calibration_img'], data['right_calibration_img'],
    #                  binocular=binocular,
    #                  file_path_2_save='./config/bicam_cal_para.json')

    print('calibration result')
    print(binocular.cam_left.camera_matrix)
    print(binocular.cam_right.camera_matrix)

    # 3d recon
    for i, (img_left_path, img_right_path) in enumerate(zip(data['left_general_img'], data['right_general_img'])):
        print('running stereo vision on ', i, img_left_path, img_right_path)
        # preprocess image
        img_left, img_right = cv2.imread(img_left_path), cv2.imread(img_right_path)
        img_left, img_right = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # extract feature
        time_start = time.time()
        if flag_reconstruct_checkboard:
            checkboard_pts_2d_left = feature.get_checkboard_corners(img_left, checkboard_size)
            checkboard_pts_2d_right = feature.get_checkboard_corners(img_right, checkboard_size)
            print('get checkboard corners in', time.time() - time_start, 'seconds')

        time_start = time.time()
        general_pts_2d_left, general_pts_2d_right = feature.get_sift_pts_pair(img_left, img_right, flag_debug=True)
        print('get features in', time.time() - time_start, 'seconds')

        # compute feature 3d coord
        if flag_reconstruct_checkboard:
            checkboard_pts_3d = binocular.transform_raw_pixel_to_world_coordiante(checkboard_pts_2d_left,
                                                                                  checkboard_pts_2d_right)
        general_pts_3d = binocular.transform_raw_pixel_to_world_coordiante(general_pts_2d_left, general_pts_2d_right)

        # statistics
        if flag_reconstruct_checkboard:
            dist = checkboard_pts_3d[:-1, :] - checkboard_pts_3d[1:, :]
            dist = np.linalg.norm(dist, axis=1)
            dist = dist[dist < square_size * 2]
            print('average distance between checkbaord corners', dist.mean())

        # vis
        mesh = o3.geometry.TriangleMesh()
        frame = mesh.create_coordinate_frame(size=25.0)

        pc_checkboard = o3.geometry.PointCloud()
        if flag_reconstruct_checkboard:
            pc_checkboard.points = o3.utility.Vector3dVector(checkboard_pts_3d)

        pc_general = o3.geometry.PointCloud()
        pc_general.points = o3.utility.Vector3dVector(general_pts_3d)

        o3.visualization.draw_geometries([pc_checkboard, pc_general, frame])
        o3.visualization.draw_geometries([pc_checkboard, pc_general])


if __name__ == '__main__':
    main()
