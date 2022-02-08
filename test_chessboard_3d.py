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
import slam_lib.geometry
import slam_lib.mapping
import slam_lib.vis
import homo


def main():
    """calibration and 3d reconstruction"""
    flag_vis_reporjection = False

    '''load data'''
    dataset_dir = '/home/cheng/Pictures/data/202201251506'
    data_stereo = slam_lib.dataset.get_calibration_and_img(dataset_dir)
    data_binocular = slam_lib.dataset.get_calibration_and_img(dataset_dir, right_img_dir_name='global')

    '''calibration'''
    stereo = BiCamera('./config/bicam_cal_para_stereo.json')
    binocular = BiCamera('./config/bicam_cal_para_binocular.json')

    checkboard_size = (11, 8)  # (board_width, board_height)
    square_size = 3.0
    # stereo = BiCamera()
    # slam_lib.camera.calibration.stereo_calibrate(square_size, checkboard_size, data_stereo['left_calibration_img'],
    #                                              data_stereo['right_calibration_img'], binocular=stereo,
    #                                              file_path_2_save='./config/bicam_cal_para_stereo.json')
    #
    # binocular = BiCamera()
    # slam_lib.camera.calibration.stereo_calibrate(square_size, checkboard_size, data_binocular['left_calibration_img'],
    #                                              data_binocular['right_calibration_img'], binocular=binocular,
    #                                              file_path_2_save='./config/bicam_cal_para_binocular.json')

    print('calibration result')
    print(stereo.cam_left.camera_matrix)
    print(stereo.cam_right.camera_matrix)

    '''3d recon'''
    for i, (img_left_path, img_right_path, img_global_path) in enumerate(zip(data_stereo['left_calibration_img'],
                                                                             data_stereo['right_calibration_img'],
                                                                             data_binocular['right_calibration_img'])):
        print('running checkboard reconstruction', i, img_left_path, img_right_path)
        # preprocess image
        img_left, img_right = cv2.imread(img_left_path), cv2.imread(img_right_path)
        gray_left, gray_right = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        img_global = cv2.imread(img_global_path)
        gray_global = cv2.cvtColor(img_global, cv2.COLOR_BGR2GRAY)

        # extract feature stereo
        time_start = time.time()
        pts_2d_left = slam_lib.feature.get_checkboard_corners(gray_left, checkboard_size, flag_vis=False)
        pts_2d_right = slam_lib.feature.get_checkboard_corners(gray_right, checkboard_size, flag_vis=False)

        print('get feature in', time.time() - time_start, 'seconds')
        # if pts_2d_left_rectified is None or pts_2d_right_rectified is None:
        #     if pts_2d_left_rectified is None: print('feature missed in left rectification')
        #     if pts_2d_right_rectified is None: print('feature missed in right rectification')
        #     continue

        # compute feature 3d coord
        pts_3d_in_left = stereo.correspondence_to_3d_in_left(pts_2d_left, pts_2d_right)

        pts_2d_reproject_in_left, *_ = cv2.projectPoints(
            pts_3d_in_left,
            rvec=np.eye(3), tvec=np.zeros(3),
            cameraMatrix=stereo.cam_left.camera_matrix,
            distCoeffs=stereo.cam_left.distortion_coefficient)

        tf_left_2_right = slam_lib.mapping.rt_2_tf(stereo.r, stereo.t)
        pts_3d_in_right = slam_lib.mapping.transform_pt_3d(tf_left_2_right, pts_3d_in_left)

        pts_2d_reproject_in_right, *_ = cv2.projectPoints(
            pts_3d_in_right,
            rvec=np.eye(3), tvec=np.zeros(3),
            cameraMatrix=stereo.cam_right.camera_matrix,
            distCoeffs=stereo.cam_right.distortion_coefficient)

        tf_left_2_global = slam_lib.mapping.rt_2_tf(binocular.r, binocular.t)
        pts_3d_in_global = slam_lib.mapping.transform_pt_3d(tf_left_2_global, pts_3d_in_left)

        pts_2d_reproject_in_global, *_ = cv2.projectPoints(
            pts_3d_in_global,
            rvec=np.eye(3), tvec=np.zeros(3),
            cameraMatrix=binocular.cam_right.camera_matrix,
            distCoeffs=binocular.cam_right.distortion_coefficient)

        # remap global img to match local img
        x_min, y_min, x_max, y_max = slam_lib.geometry.corners_2_bounding_box_xyxy(pts_2d_reproject_in_global[:, 0, :])
        h, w = y_max - y_min, x_max - x_min
        range_x_min, range_y_min, range_x_max, range_y_max = x_min - int(w/2), y_min - int(h/2), x_max + int(w/2), y_max + int(h/2)
        box_x_min, box_y_min, box_x_max, box_y_max = x_min - range_x_min, y_min - range_y_min, x_max - range_x_min, y_max - range_y_min

        img_left_in_global = img_global[range_y_min:range_y_max, range_x_min:range_x_max, :]
        # img_left_object = cv2.resize(img_left, dsize=(img_left_in_global.shape[1], img_left_in_global.shape[0]))
        img_left_object = np.copy(img_left)

        # map_ = np.eye(3)
        # map_[0, 0] =
        # map_[1, 1] =
        # map_[0, -1] =
        # map_[1, -1] =

        # find correspondence and homo
        homography = homo.find_homo(img_left_in_global, img_left_object, flag_vis_feature_matching=True)
        if homography is None:
            print('no homo found')
            homography = np.eye(3)

        # map object in global img coord to local img coord
        # object_pts_2d_global = np.array([[x_min, y_min],
        #                                  [x_max, y_max]]).astype(int)

        object_pts_2d_global = pts_2d_reproject_in_global[:, 0, :].astype(int)
        object_pts_2d_global[:, 0] = object_pts_2d_global[:, 0] - range_x_min
        object_pts_2d_global[:, 1] = object_pts_2d_global[:, 1] - range_y_min
        object_pts_2d_left = slam_lib.mapping.transform_pt_2d(homography, object_pts_2d_global).astype(int)

        print(homography)
        print(object_pts_2d_global)
        print(object_pts_2d_left)


        # vis skin points in local and global
        # draw match
        if flag_vis_reporjection:
            end = 5
            img_match = slam_lib.vis.draw_matches(gray_left, pts_2d_left[:end], gray_right, pts_2d_right[:end])
            cv2.namedWindow('checkboard points match', cv2.WINDOW_NORMAL)
            cv2.imshow('checkboard points match', img_match)
            cv2.waitKey(0)

            # draw reprojection
            img_left_feature = np.copy(img_left)
            for pts_2d in pts_2d_reproject_in_left[:, 0, :]:
                pts_2d = pts_2d.astype(int)
                img_left_feature = cv2.circle(img_left_feature, center=pts_2d, radius=15, color=(0, 0, 255), thickness=5)
            cv2.namedWindow('hair skin pt in global img', cv2.WINDOW_NORMAL)
            cv2.imshow('hair skin pt in global img', img_left_feature)
            cv2.waitKey(0)

            img_right_feature = np.copy(img_right)
            for pts_2d in pts_2d_reproject_in_right[:, 0, :]:
                pts_2d = pts_2d.astype(int)
                img_right_feature = cv2.circle(img_right_feature, center=pts_2d, radius=15, color=(0, 0, 255), thickness=5)
            cv2.namedWindow('hair skin pt in global img', cv2.WINDOW_NORMAL)
            cv2.imshow('hair skin pt in global img', img_right_feature)
            cv2.waitKey(0)

        img_global_feature = np.copy(img_global)
        for pts_2d in pts_2d_reproject_in_global[:, 0, :]:
            pts_2d = pts_2d.astype(int)
            img_global_feature = cv2.circle(img_global_feature, center=pts_2d, radius=4, color=(0, 0, 255), thickness=2)
        cv2.namedWindow('pts in global img', cv2.WINDOW_NORMAL)
        cv2.imshow('pts in global img', img_global_feature)
        cv2.waitKey(0)

        img_global_object = np.copy(img_left_in_global)
        for pts_2d in object_pts_2d_global:
            pts_2d = pts_2d.astype(int)
            img_global_object = cv2.circle(img_global_object, center=pts_2d, radius=4, color=(0, 0, 255), thickness=2)
        cv2.namedWindow('pts in global img patch', cv2.WINDOW_NORMAL)
        cv2.imshow('pts in global img patch', img_global_object)
        cv2.waitKey(0)

        img_left_object = np.copy(img_left)
        for pts_2d in object_pts_2d_left:
            pts_2d = pts_2d.astype(int)
            img_left_object = cv2.circle(img_left_object, center=pts_2d, radius=15, color=(0, 0, 255), thickness=2)
        cv2.namedWindow('pts in left img', cv2.WINDOW_NORMAL)
        cv2.imshow('pts in left img', img_left_object)
        cv2.waitKey(0)

        # img_global_object = cv2.rectangle(img_left_in_global, pt1=object_pts_2d_global[0], pt2=object_pts_2d_global[1], color=(0, 0, 255), thickness=5)
        # cv2.namedWindow('hair skin pt in global img', cv2.WINDOW_NORMAL)
        # cv2.imshow('hair skin pt in global img', img_global_object)
        # cv2.waitKey(0)

        # statistics
        dist = pts_3d_in_left[:-1, :] - pts_3d_in_left[1:, :]
        dist = np.linalg.norm(dist, axis=1)
        dist = dist[dist < square_size * 2]
        print('average distance between checkbaord corners', dist.mean())

        # vis left
        mesh = o3.geometry.TriangleMesh()
        frame_left = mesh.create_coordinate_frame(size=25.0)
        frame_right = mesh.create_coordinate_frame(size=25.0)
        frame_right.transform(np.linalg.inv(tf_left_2_right))
        frame_global = mesh.create_coordinate_frame(size=55.0)
        frame_global.transform(np.linalg.inv(tf_left_2_global))

        pc_general_left = o3.geometry.PointCloud()
        pc_general_left.points = o3.utility.Vector3dVector(pts_3d_in_left)

        pc_general_right = o3.geometry.PointCloud()
        # pc_general_right.points = o3.utility.Vector3dVector(mapping.transform_pt_3d(tf_left_2_right, pts_3d_in_left))

        # pc_rect = o3.geometry.PointCloud()
        # pc_rect.points = o3.utility.Vector3dVector(pts_3d_from_rect)

        o3.visualization.draw_geometries([pc_general_left, frame_left, frame_right, frame_global])
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
