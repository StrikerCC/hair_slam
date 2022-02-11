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
from slam_lib.camera.cam import StereoCamera
from slam_lib.camera.calibration import stereo_calibrate
import slam_lib.geometry
import slam_lib.mapping
import slam_lib.vis
import homo


def get_stereo_calibration(square_size, checkboard_size, data_stereo, calibration_parameter_saving_file_path,
                           calbration=False):
    """

    :return:
    """

    '''calibration'''
    stereo_cal_status = False
    stereo = StereoCamera()

    if calbration:
        slam_lib.camera.calibration.stereo_calibrate(square_size, checkboard_size, data_stereo['left_calibration_img'],
                                                     data_stereo['right_calibration_img'], binocular=stereo,
                                                     file_path_2_save='./config/bicam_cal_para_stereo.json')
        stereo_cal_status = True
    else:
        stereo_cal_status = stereo.read_cal_param_file_and_set_params(calibration_parameter_saving_file_path)
    if stereo_cal_status:
        print('calibration result saved at', calibration_parameter_saving_file_path)
        print(stereo.cam_left.camera_matrix)
        print(stereo.cam_right.camera_matrix)
    else:
        Warning('stereo not set')
        Warning('stereo not set')
        Warning('stereo not set')
    return stereo


def test_match_by_projection(stereo, img_left, img_right, pts_2d_left_raw, pts_2d_right_raw, pts_3d_in_left,
                             pts_3d_in_right):
    """"""
    tf_left_2_right = slam_lib.mapping.rt_2_tf(stereo.r, stereo.t)

    '''find interested pts correspondence between left and right img'''
    # get interested pt as checkboard corners
    # interested_pts_2d_left = pts_2d_left_raw
    # interested_pts_2d_right = pts_2d_right_raw

    # get interested pt by sift points
    interested_pts_2d_left_raw, des1, interested_pts_2d_right_raw, des2 = \
        slam_lib.feature.get_sift_pts_and_sift_feats(img_left, img_right, shrink=4.5, flag_debug=True)

    # bf = cv2.BFMatcher_create()
    # matches = bf.knnMatch(des1, des2, k=2)
    # good_matches = [first for first, second in matches if first.distance < 0.85 * second.distance]
    # index_match = np.asarray([[m.queryIdx, m.trainIdx] for m in good_matches])
    #
    # interested_pts_2d_left = interested_pts_2d_left[index_match[:, 0].tolist()]  # queryIdx
    # des1 = des1[index_match[:, 0]]
    # interested_pts_2d_right = interested_pts_2d_right[index_match[:, 1].tolist()]  # trainIdx
    # des2 = des2[index_match[:, 1]]

    interested_pts_2d_left = cv2.undistortPoints(interested_pts_2d_left_raw, cameraMatrix=stereo.cam_left.camera_matrix,
                                                 distCoeffs=stereo.cam_left.distortion_coefficient)
    interested_pts_2d_left = interested_pts_2d_left[:, 0, :]

    interested_pts_2d_right = cv2.undistortPoints(interested_pts_2d_right_raw,
                                                  cameraMatrix=stereo.cam_right.camera_matrix,
                                                  distCoeffs=stereo.cam_right.distortion_coefficient)
    interested_pts_2d_right = interested_pts_2d_right[:, 0, :]

    # compute project line, which is interested pts on image
    interested_lines_3d_left_img_plane = np.concatenate(
        [np.copy(interested_pts_2d_left), np.ones((interested_pts_2d_left.shape[0], 1))],
        axis=-1) * 300
    interested_lines_3d_right_img_plane = np.concatenate(
        [np.copy(interested_pts_2d_right), np.ones((interested_pts_2d_right.shape[0], 1))],
        axis=-1) * 300

    # find 3d points closeted to project line
    index_left_lines_2_pts_3d_left, interested_lines_3d_left_img_plane_normalized = slam_lib.geometry.nearest_points_2_lines(
        interested_lines_3d_left_img_plane, pts_3d_in_left)
    index_right_lines_2_pts_3d_right, interested_lines_3d_right_img_plane_normalized = slam_lib.geometry.nearest_points_2_lines(
        interested_lines_3d_right_img_plane, pts_3d_in_right)

    # get interested point project on ideal surface formed by 3d points
    ## get nearest 3d points for each line
    pts_3d_in_left_close_to_interested_pts = pts_3d_in_left[index_left_lines_2_pts_3d_left]
    pts_3d_in_right_close_to_interested_pts = pts_3d_in_right[index_right_lines_2_pts_3d_right]

    interested_pts_3d_left_proj_in_left = slam_lib.geometry.closet_vector_2_point(
        interested_lines_3d_left_img_plane_normalized,
        pts_3d_in_left_close_to_interested_pts)
    interested_pts_3d_right_proj_in_right = slam_lib.geometry.closet_vector_2_point(
        interested_lines_3d_right_img_plane_normalized,
        pts_3d_in_right_close_to_interested_pts)
    interested_pts_3d_right_proj_in_left = slam_lib.mapping.transform_pt_3d(np.linalg.inv(tf_left_2_right),
                                                                            interested_pts_3d_right_proj_in_right)

    # pc_general_left = o3.geometry.PointCloud()
    # pc_general_left.points = o3.utility.Vector3dVector(pts_3d_in_left)

    # nn search to match interested point project on ideal surface
    id_left_interested_pts_2_right_interested_pts = slam_lib.geometry.nearest_neighbor_points_2_points(
        interested_pts_3d_left_proj_in_left, interested_pts_3d_right_proj_in_left)
    pts_2d_raw_left_match, pts_2d_raw_right_match = [], []
    for id_left_interested_pt, id_right_interested_pt in id_left_interested_pts_2_right_interested_pts:
        pts_2d_raw_left_match.append(interested_pts_2d_left_raw[id_left_interested_pt])
        pts_2d_raw_right_match.append(interested_pts_2d_right_raw[id_right_interested_pt])
    pts_2d_raw_left_match = np.asarray(pts_2d_raw_left_match)
    pts_2d_raw_right_match = np.asarray(pts_2d_raw_right_match)

    # vis
    ## draw match
    end = -1
    img_match = slam_lib.vis.draw_matches(img_left, pts_2d_raw_left_match[:end], img_right,
                                          pts_2d_raw_right_match[:end])
    cv2.namedWindow('checkboard points reprojection match', cv2.WINDOW_NORMAL)
    cv2.imshow('checkboard points reprojection match', img_match)
    cv2.waitKey(0)

    # lines
    origins = np.zeros(interested_lines_3d_left_img_plane.shape)
    lines_left = o3.geometry.LineSet()
    lines_left.points = o3.utility.Vector3dVector(np.vstack([interested_lines_3d_left_img_plane, origins]))
    lines_left.lines = o3.utility.Vector2iVector(
        np.arange(0, 2 * len(interested_lines_3d_left_img_plane)).reshape(2, -1).T)
    colors = [[1, 0, 0] for i in range(2 * len(interested_lines_3d_left_img_plane))]
    lines_left.colors = o3.utility.Vector3dVector(colors)

    lines_right = o3.geometry.LineSet()
    lines_right.points = o3.utility.Vector3dVector(np.vstack([interested_lines_3d_right_img_plane, origins]))
    lines_right.lines = o3.utility.Vector2iVector(
        np.arange(0, 2 * len(interested_lines_3d_right_img_plane)).reshape(2, -1).T)
    colors = [[200, 0, 0] for i in range(2 * len(interested_lines_3d_right_img_plane))]
    lines_right.colors = o3.utility.Vector3dVector(colors)
    lines_right.transform(np.linalg.inv(tf_left_2_right))

    # pc
    pc_general_left = o3.geometry.PointCloud()
    pc_general_left.points = o3.utility.Vector3dVector(pts_3d_in_left)

    pc_general_right = o3.geometry.PointCloud()
    pc_general_right.points = o3.utility.Vector3dVector(pts_3d_in_right)
    pc_general_right.transform(np.linalg.inv(tf_left_2_right))

    pc_interested = o3.geometry.PointCloud()
    pc_interested.points = o3.utility.Vector3dVector(interested_pts_3d_left_proj_in_left)

    # frames
    mesh = o3.geometry.TriangleMesh()
    frame_left = mesh.create_coordinate_frame(size=20.0)
    frame_right = mesh.create_coordinate_frame(size=20.0)
    frame_right.transform(np.linalg.inv(tf_left_2_right))

    o3.visualization.draw_geometries(
        [frame_left, frame_right, lines_left, pc_general_left, lines_right, pc_general_right, pc_interested])

    pts_2d_left_img_plane_reproject, *_ = cv2.projectPoints(
        interested_lines_3d_left_img_plane_normalized, rvec=np.eye(3), tvec=np.zeros(3),
        cameraMatrix=stereo.cam_left.camera_matrix,
        distCoeffs=stereo.cam_right.distortion_coefficient)
    pts_2d_left_img_plane_reproject = pts_2d_left_img_plane_reproject[:, 0, :]

    img_left_object = np.copy(img_left)
    for pts_2d in pts_2d_left_img_plane_reproject:
        pts_2d = pts_2d.astype(int)
        img_left_object = cv2.circle(img_left_object, center=pts_2d, radius=15, color=(0, 0, 255), thickness=2)
    cv2.namedWindow('pts in left img', cv2.WINDOW_NORMAL)
    cv2.imshow('pts in left img', img_left_object)
    cv2.waitKey(0)

    # match 3d points with nn search


def test_global_2_local_homo(stereo, binocular, img_left, img_right, img_global, pts_2d_left_raw, pts_2d_right_raw,
                             pts_3d_in_left, pts_3d_in_right, pts_3d_in_global):
    flag_vis_porjection = True
    flag_vis_global_2_local_mapping = True

    tf_left_2_right = slam_lib.mapping.rt_2_tf(stereo.r, stereo.t)
    tf_left_2_global = slam_lib.mapping.rt_2_tf(binocular.r, binocular.t)
    pts_3d_in_global = slam_lib.mapping.transform_pt_3d(tf_left_2_global, pts_3d_in_left)

    pts_2d_reproject_in_left, *_ = cv2.projectPoints(
        pts_3d_in_left,
        rvec=np.eye(3), tvec=np.zeros(3),
        cameraMatrix=stereo.cam_left.camera_matrix,
        distCoeffs=stereo.cam_left.distortion_coefficient)
    pts_2d_reproject_in_left = pts_2d_reproject_in_left[:, 0, :]

    pts_2d_reproject_in_right, *_ = cv2.projectPoints(
        pts_3d_in_right,
        rvec=np.eye(3), tvec=np.zeros(3),
        cameraMatrix=stereo.cam_right.camera_matrix,
        distCoeffs=stereo.cam_right.distortion_coefficient)
    pts_2d_reproject_in_right = pts_2d_reproject_in_right[:, 0, :]

    pts_2d_reproject_in_global, *_ = cv2.projectPoints(
        pts_3d_in_global,
        rvec=np.eye(3), tvec=np.zeros(3),
        cameraMatrix=binocular.cam_right.camera_matrix,
        distCoeffs=binocular.cam_right.distortion_coefficient)
    pts_2d_reproject_in_global = pts_2d_reproject_in_global[:, 0, :]

    '''map object in global img to local img'''
    # find global patch to match local
    x_min, y_min, x_max, y_max = slam_lib.geometry.corners_2_bounding_box_xyxy(pts_2d_reproject_in_global)
    h, w = y_max - y_min, x_max - x_min
    range_x_min, range_y_min, range_x_max, range_y_max = x_min - int(w / 2), y_min - int(h / 2), x_max + int(
        w / 2), y_max + int(h / 2)

    # img_patch_global_2_match_left = img_global[range_y_min:range_y_max, range_x_min:range_x_max, :]
    #
    # # find correspondence and homo between global patch and local
    # homography_global_2_global_patch = np.array([[1, 0, -range_x_min], [0, 1, -range_y_min], [0, 0, 1]])
    # homography_global_patch_2_left = homo.find_homo_between_two_imgs(img_patch_global_2_match_left, img_left,
    #                                                                  flag_vis_feature_matching=False)
    # if homography_global_patch_2_left is None:
    #     print('no homo found')
    #     homography_global_patch_2_left = np.eye(3)
    #
    # homography_global_2_left = np.matmul(homography_global_patch_2_left, homography_global_2_global_patch)
    #
    # # map object in global img coord to local img coord
    # object_pts_2d_global = pts_2d_reproject_in_global
    # object_pts_2d_left = slam_lib.mapping.transform_pt_2d(homography_global_2_left, object_pts_2d_global)
    #
    # # map local field of view to global
    # field_of_view_left = np.array([[0, 0],
    #                                [0, img_left.shape[0]],
    #                                [img_left.shape[1], img_left.shape[0]],
    #                                [img_left.shape[1], 0]])
    # field_of_view_left = np.concatenate([field_of_view_left / 2, field_of_view_left], axis=0)  # make upper left square
    # field_of_view_left_in_global = slam_lib.mapping.transform_pt_2d(np.linalg.inv(homography_global_2_left),
    #                                                                 field_of_view_left).astype(int)
    # print(homography_global_patch_2_left)
    # print(object_pts_2d_global)
    # print(object_pts_2d_left)

    # vis skin points in local and global
    # draw match
    if flag_vis_porjection:
        end = 8
        img_match = slam_lib.vis.draw_matches(img_left, pts_2d_left_raw[:end], img_right, pts_2d_right_raw[:end])
        cv2.namedWindow('checkboard points match', cv2.WINDOW_NORMAL)
        cv2.imshow('checkboard points match', img_match)
        cv2.waitKey(0)

        # draw reprojection
        img_left_proj_checkboard = np.copy(img_left)
        for pts_2d in pts_2d_reproject_in_left:
            pts_2d = pts_2d.astype(int)
            img_left_proj_checkboard = cv2.circle(img_left_proj_checkboard, center=pts_2d, radius=15, color=(0, 0, 255),
                                                  thickness=5)
        cv2.namedWindow('3d checkboard project left img', cv2.WINDOW_NORMAL)
        cv2.imshow('3d checkboard project left img', img_left_proj_checkboard)
        cv2.waitKey(0)

        img_righ_proj_checkboard = np.copy(img_right)
        for pts_2d in pts_2d_reproject_in_right:
            pts_2d = pts_2d.astype(int)
            img_righ_proj_checkboard = cv2.circle(img_righ_proj_checkboard, center=pts_2d, radius=15, color=(0, 0, 255),
                                                  thickness=5)
        cv2.namedWindow('3d checkboard project right img', cv2.WINDOW_NORMAL)
        cv2.imshow('3d checkboard project right img', img_righ_proj_checkboard)
        cv2.waitKey(0)

        img_global_proj_checkboard = np.copy(img_global)
        for pts_2d in pts_2d_reproject_in_global:
            pts_2d = pts_2d.astype(int)
            img_global_proj_checkboard = cv2.circle(img_global_proj_checkboard, center=pts_2d, radius=4,
                                                    color=(0, 0, 255), thickness=2)
        cv2.namedWindow('3d checkboard project in global img', cv2.WINDOW_NORMAL)
        cv2.imshow('3d checkboard project in global img', img_global_proj_checkboard)
        cv2.waitKey(0)

    # if flag_vis_global_2_local_mapping:
    #     img_global_mark_object = np.copy(img_global)
    #     for pts_2d in object_pts_2d_global:
    #         pts_2d = pts_2d.astype(int)
    #         img_global_mark_object = cv2.circle(img_global_mark_object, center=pts_2d, radius=4, color=(0, 0, 255),
    #                                             thickness=2)
    #     cv2.namedWindow('object pts in global img', cv2.WINDOW_NORMAL)
    #     cv2.imshow('object pts in global img', img_global_mark_object)
    #     cv2.waitKey(0)
    #
    #     img_left_mark_object = np.copy(img_left)
    #     for pts_2d in object_pts_2d_left.astype(int):
    #         pts_2d = pts_2d.astype(int)
    #         img_left_mark_object = cv2.circle(img_left_mark_object, center=pts_2d, radius=15, color=(0, 0, 255),
    #                                           thickness=2)
    #     cv2.namedWindow('object pts in left img', cv2.WINDOW_NORMAL)
    #     cv2.imshow('object pts in left img', img_left_mark_object)
    #     cv2.waitKey(0)
    #
    #     img_left_field_of_view = np.copy(img_left)
    #     for pts_2d_start, pts_2d_end in zip(field_of_view_left,
    #                                         np.concatenate([field_of_view_left[1:], field_of_view_left[0:1]], axis=0)):
    #         pts_2d_start = pts_2d_start.astype(int)
    #         pts_2d_end = pts_2d_end.astype(int)
    #         img_left_field_of_view = cv2.line(img_left_field_of_view, pt1=pts_2d_start, pt2=pts_2d_end,
    #                                           color=(0, 0, 255), thickness=2)
    #     cv2.namedWindow('half field of view in left img', cv2.WINDOW_NORMAL)
    #     cv2.imshow('half field of view in left img', img_left_field_of_view)
    #     cv2.waitKey(0)
    #
    #     img_left_field_of_view_in_global = np.copy(img_global)
    #     for pts_2d_start, pts_2d_end in zip(field_of_view_left_in_global, np.concatenate(
    #             [field_of_view_left_in_global[1:], field_of_view_left_in_global[0:1]], axis=0)):
    #         pts_2d_start = pts_2d_start.astype(int)
    #         pts_2d_end = pts_2d_end.astype(int)
    #         img_left_field_of_view_in_global = cv2.line(img_left_field_of_view_in_global, pt1=pts_2d_start,
    #                                                     pt2=pts_2d_end, color=(0, 0, 255), thickness=2)
    #     cv2.namedWindow('left field of view in global img', cv2.WINDOW_NORMAL)
    #     cv2.imshow('left field of view in global img', img_left_field_of_view_in_global)
    #     cv2.waitKey(0)


def main():
    """calibration and 3d reconstruction"""

    '''load data'''
    checkboard_size = (11, 8)  # (board_width, board_height)
    square_size = 3.0
    dataset_dir = '/home/cheng/Pictures/data/202201251506'
    data_stereo = slam_lib.dataset.get_calibration_and_img(dataset_dir)
    data_binocular = slam_lib.dataset.get_calibration_and_img(dataset_dir, right_img_dir_name='global')
    stereo = get_stereo_calibration(square_size, checkboard_size, data_stereo,
                                    calibration_parameter_saving_file_path='./config/bicam_cal_para_stereo.json')
    binocular = get_stereo_calibration(square_size, checkboard_size, data_binocular,
                                       calibration_parameter_saving_file_path='./config/bicam_cal_para_binocular.json')

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
        tf_left_2_right = slam_lib.mapping.rt_2_tf(stereo.r, stereo.t)
        tf_left_2_global = slam_lib.mapping.rt_2_tf(binocular.r, binocular.t)

        # extract feature from stereo
        time_start = time.time()
        pts_2d_left_raw = slam_lib.feature.get_checkboard_corners(gray_left, checkboard_size, flag_vis=False)
        pts_2d_right_raw = slam_lib.feature.get_checkboard_corners(gray_right, checkboard_size, flag_vis=False)

        pts_2d_left = cv2.undistortPoints(pts_2d_left_raw, cameraMatrix=stereo.cam_left.camera_matrix,
                                          distCoeffs=stereo.cam_left.distortion_coefficient)
        pts_2d_left = pts_2d_left[:, 0, :]

        pts_2d_right = cv2.undistortPoints(pts_2d_right_raw, cameraMatrix=stereo.cam_right.camera_matrix,
                                           distCoeffs=stereo.cam_right.distortion_coefficient)
        pts_2d_right = pts_2d_right[:, 0, :]

        print('get checkboard corners in', time.time() - time_start, 'seconds')
        # if pts_2d_left_rectified is None or pts_2d_right_rectified is None:
        #     if pts_2d_left_rectified is None: print('feature missed in left rectification')
        #     if pts_2d_right_rectified is None: print('feature missed in right rectification')
        #     continue

        # compute feature 3d coord
        pts_3d_in_left = stereo.correspondence_to_3d_in_left(pts_2d_left_raw, pts_2d_right_raw)
        pts_3d_in_right = slam_lib.mapping.transform_pt_3d(tf_left_2_right, pts_3d_in_left)
        pts_3d_in_global = slam_lib.mapping.transform_pt_3d(tf_left_2_global, pts_3d_in_left)

        '''testing functionalities'''
        # test_match_by_projection(stereo, img_left, img_right, pts_2d_left_raw, pts_2d_right_raw, pts_3d_in_left,
        #                          pts_3d_in_right)

        test_global_2_local_homo(stereo, binocular, img_left, img_right, img_global, pts_2d_left_raw, pts_2d_right_raw,
                                 pts_3d_in_left, pts_3d_in_right, pts_3d_in_global)

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
