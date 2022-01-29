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

import slam_lib.dataset
import slam_lib.feature
from slam_lib.camera.cam import BiCamera
from slam_lib.camera.calibration import stereo_calibrate
import slam_lib.mapping
import slam_lib.vis


def main():
    """calibration and 3d reconstruction"""

    # cal
    square_size = 3.0
    checkboard_size = (11, 8)  # (board_width, board_height)

    dataset_dir = '/home/cheng/Pictures/data/202201251506'
    data = slam_lib.dataset.get_calibration_and_img(dataset_dir, right_img_dir_name='global')
    stereo = BiCamera('./config/bicam_cal_para.json')

    # stereo = BiCamera()
    # stereo_calibrate(square_size, checkboard_size, data['left_calibration_img'], data['right_calibration_img'], binocular=stereo,
    #                  file_path_2_save='./config/bicam_cal_para.json')

    # binocular = BiCamera('./config/bicam_cal_para.json')

    print('calibration result')
    print(stereo.cam_left.camera_matrix)
    print(stereo.cam_right.camera_matrix)

    # 3d recon
    for i, (img_left_path, img_right_path) in enumerate(zip(data['left_calibration_img'], data['right_calibration_img'])):
        print('running checkboard reconstruction', i, img_left_path, img_right_path)
        # preprocess image
        img_left, img_right = cv2.imread(img_left_path), cv2.imread(img_right_path)
        gray_left, gray_right = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        # img_left_rectified, img_right_rectified = binocular.cam_left.undistort_rectify_img(
        #     img_left), binocular.cam_right.undistort_rectify_img(img_right)

        # extract feature
        time_start = time.time()
        pts_2d_left = slam_lib.feature.get_checkboard_corners(gray_left, checkboard_size, flag_vis=False)
        pts_2d_right = slam_lib.feature.get_checkboard_corners(gray_right, checkboard_size, flag_vis=False)

        # pts_2d_left_rectified = feature.get_checkboard_corners(img_left_rectified, checkboard_size)
        # pts_2d_right_rectified = feature.get_checkboard_corners(img_right_rectified, checkboard_size)

        # print('get feature in', time.time() - time_start, 'seconds')
        # if pts_2d_left_rectified is None or pts_2d_right_rectified is None:
        #     if pts_2d_left_rectified is None: print('feature missed in left rectification')
        #     if pts_2d_right_rectified is None: print('feature missed in right rectification')
        #     continue
        # compute feature 3d coord
        pts_3d_in_left_rectify = stereo.transform_raw_pixel_to_world_coordiante(pts_2d_left, pts_2d_right)
        # pts_3d_from_rect = binocular.transform_rectify_pixel_to_world_coordiante(pts_2d_left_rectified,
        #                                                                          pts_2d_right_rectified)

        # pts_3d_general_skin_in_right = utils.mapping.transform_pt_3d(tf=tf_left_2_right, pts=pts_3d_general_skin_in_left)

        # compute 2d pixel coord of interested 3d pts in global pixel frame
        tf_left_2_left_rectify = slam_lib.mapping.rt_2_tf(stereo.cam_left.rotation_rectify, np.zeros((3, 1)))
        pts_3d_in_left = slam_lib.mapping.transform_pt_3d(np.linalg.inv(tf_left_2_left_rectify), pts_3d_in_left_rectify)

        pts_2d_reproject_in_left, *_ = cv2.projectPoints(
            pts_3d_in_left,
            rvec=np.eye(3), tvec=np.zeros(3),
            cameraMatrix=stereo.cam_left.camera_matrix,
            distCoeffs=stereo.cam_left.distortion_coefficient)

        tf_left_2_right = slam_lib.mapping.rt_2_tf(stereo.r, stereo.t)
        # tf_right_2_right_rectify = mapping.rt_2_tf(stereo.cam_right.rotation_rectify, np.zeros((3, 1)))     # tf from world to rectified camera, then from rectified camera to un-rectified camera
        pts_3d_in_right = slam_lib.mapping.transform_pt_3d(tf_left_2_right, pts_3d_in_left)

        pts_2d_reproject_in_right, *_ = cv2.projectPoints(
            pts_3d_in_right,
            rvec=np.eye(3), tvec=np.zeros(3),
            cameraMatrix=stereo.cam_right.camera_matrix,
            distCoeffs=stereo.cam_right.distortion_coefficient)

        # pts_2d_general_skin_in_right = stereo.cam_right.proj_and_distort(pts_3d_general_skin_in_right) # projection

        # vis skin points in local and global
        # draw match
        end = 5
        img_match = slam_lib.vis.draw_matches(gray_left, pts_2d_left[:end], gray_right, pts_2d_right[:end])
        cv2.namedWindow('checkboard points match', cv2.WINDOW_NORMAL)
        cv2.imshow('checkboard points match', img_match)
        cv2.waitKey(0)

        # draw reprojection
        for pts_2d in pts_2d_reproject_in_left[:, 0, :]:
            pts_2d = pts_2d.astype(int)
            img_left = cv2.circle(img_left, center=pts_2d, radius=15, color=(0, 0, 255), thickness=5)
        cv2.namedWindow('hair skin pt in global img', cv2.WINDOW_NORMAL)
        cv2.imshow('hair skin pt in global img', img_left)
        cv2.waitKey(0)

        for pts_2d in pts_2d_reproject_in_right[:, 0, :]:
            pts_2d = pts_2d.astype(int)
            img_right = cv2.circle(img_right, center=pts_2d, radius=15, color=(0, 0, 255), thickness=5)
        cv2.namedWindow('hair skin pt in global img', cv2.WINDOW_NORMAL)
        cv2.imshow('hair skin pt in global img', img_right)
        cv2.waitKey(0)

        # statistics
        dist = pts_3d_in_left_rectify[:-1, :] - pts_3d_in_left_rectify[1:, :]
        dist = np.linalg.norm(dist, axis=1)
        dist = dist[dist < square_size*2]
        print('average distance between checkbaord corners', dist.mean())

        # vis left
        mesh = o3.geometry.TriangleMesh()
        frame_left = mesh.create_coordinate_frame(size=25.0)
        frame_right = mesh.create_coordinate_frame(size=45.0)
        frame_right.transform(np.linalg.inv(tf_left_2_right))

        pc_general_left = o3.geometry.PointCloud()
        # pc_general_left.points = o3.utility.Vector3dVector(pts_3d_in_left)

        pc_general_right = o3.geometry.PointCloud()
        # pc_general_right.points = o3.utility.Vector3dVector(mapping.transform_pt_3d(tf_left_2_right, pts_3d_in_left))

        # pc_rect = o3.geometry.PointCloud()
        # pc_rect.points = o3.utility.Vector3dVector(pts_3d_from_rect)

        o3.visualization.draw_geometries([pc_general_left, frame_left, frame_right])
        # o3.visualization.draw_geometries([pc_raw, pc_rect, frame])
        # o3.visualization.draw_geometries([pc_raw, pc_rect])


        '''vis right main'''
        frame_left = mesh.create_coordinate_frame(size=25.0)
        # frame_left.transform()
        # frame_right = mesh.create_coordinate_frame(size=45.0)

        # mask out some wrong point by distance away from camera
        # pts_3d = pts_3d[np.linalg.norm(pts_3d, axis=-1) < distance_max]
        # pc_general.points = o3.utility.Vector3dVector(pts_3d_in_right)
        #
        # o3.visualization.draw_geometries([pc_general, frame_left, frame_right])
        # o3.visualization.draw_geometries([pc_general])


if __name__ == '__main__':
    main()
