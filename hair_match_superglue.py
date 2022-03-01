# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/31/21 4:43 PM
"""
import time

import os
import cv2
import numpy as np
import open3d as o3

import slam_lib.dataset
import slam_lib.feature
from slam_lib.camera.cam import StereoCamera
from slam_lib.camera.calibration import stereo_calibrate

import slam_lib.format
import slam_lib.geometry
import slam_lib.mapping
import slam_lib.vis


def match_by_projection(stereo, img_left, left_interested_pts_2d_left_img, pts_3d_in_left,
                        img_right, right_interested_pts_2d_right_img, pts_3d_in_right, flag_debug=True):
    """"""

    timer = slam_lib.format.timer()

    normal_dis_to_recon = 1.0                       # mm
    interesting_pt_3drecon_distance_error = 1.0     # mm
    epipolar_error = 5.0                            # pixels

    tf_left_2_right = slam_lib.mapping.rt_2_tf(stereo.r, stereo.t)

    '''img frame to cm frame'''
    left_interested_pts_2d_left_cam = stereo.cam_left.proj_pts_2d_img_frame_2_new_frame(left_interested_pts_2d_left_img)
    right_interested_pts_2d_right_cam = stereo.cam_left.proj_pts_2d_img_frame_2_new_frame(right_interested_pts_2d_right_img)

    '''compute project line, which is interested pts on image'''
    left_interested_lines_3d_left_cam = slam_lib.format.pts_2d_2_3d_homo(left_interested_pts_2d_left_cam) * 300
    right_interested_lines_3d_right_cam = slam_lib.format.pts_2d_2_3d_homo(right_interested_pts_2d_right_cam) * 300

    '''find 3d points closeted to project line'''
    index_lines_2_pts_3d_left, left_interested_lines_3d_right_cam_normalized = slam_lib.geometry.nearest_points_2_lines(
        left_interested_lines_3d_left_cam, pts_3d_in_left, normal_distance_min=normal_dis_to_recon)
    index_lines_2_pts_3d_right, right_interested_lines_3d_right_cam_normalized = slam_lib.geometry.nearest_points_2_lines(
        right_interested_lines_3d_right_cam, pts_3d_in_right, normal_distance_min=normal_dis_to_recon)

    '''get interested point project on ideal surface formed by 3d points'''
    # get nearest 3d points for each line
    pts_3d_in_left_closet_to_interested_pts = pts_3d_in_left[index_lines_2_pts_3d_left]
    pts_3d_in_right_closet_to_interested_pts = pts_3d_in_right[index_lines_2_pts_3d_right]

    left_interested_pts_3d_left_cam = slam_lib.geometry.closet_vector_2_point(
        left_interested_lines_3d_right_cam_normalized,
        pts_3d_in_left_closet_to_interested_pts)
    right_interested_pts_3d_right_cam = slam_lib.geometry.closet_vector_2_point(
        right_interested_lines_3d_right_cam_normalized,
        pts_3d_in_right_closet_to_interested_pts)
    right_interested_pts_3d_left_cam = slam_lib.mapping.transform_pt_3d(np.linalg.inv(tf_left_2_right),
                                                                        right_interested_pts_3d_right_cam)

    '''nn search to match interested point'''
    id_left_interested_pts_2_right_interested_pts = \
        slam_lib.geometry.nearest_neighbor_points_2_points(left_interested_pts_3d_left_cam,
                                                           right_interested_pts_3d_left_cam, distance_min=interesting_pt_3drecon_distance_error)
    left_interested_pts_2d_left_img_match, right_interested_pts_2d_right_img_match = \
        left_interested_pts_2d_left_img[id_left_interested_pts_2_right_interested_pts[:, 0].tolist()], \
        right_interested_pts_2d_right_img[id_left_interested_pts_2_right_interested_pts[:, 1].tolist()]

    '''filter by epipolar geometry'''
    interested_pts_2d_left_cam_rectified_match, interested_pts_2d_right_cam_rectified_match = \
        stereo.cam_left.proj_pts_2d_img_frame_2_new_frame(left_interested_pts_2d_left_img_match,
                                                          rotation=stereo.rotation_rectify_left,
                                                          camera_matrix=stereo.camera_matrix_rectify_left), \
        stereo.cam_right.proj_pts_2d_img_frame_2_new_frame(right_interested_pts_2d_right_img_match,
                                                           rotation=stereo.rotation_rectify_right,
                                                           camera_matrix=stereo.camera_matrix_rectify_right)
    mask_epipolar = np.abs(interested_pts_2d_left_cam_rectified_match[:, 1] - interested_pts_2d_right_cam_rectified_match[:, 1]) < epipolar_error
    id_left_interested_pts_2_right_interested_pts = id_left_interested_pts_2_right_interested_pts[mask_epipolar]

    interested_pts_2d_left_img_match, interested_pts_2d_right_img_match = \
        left_interested_pts_2d_left_img[id_left_interested_pts_2_right_interested_pts[:, 0].tolist()], \
        right_interested_pts_2d_right_img[id_left_interested_pts_2_right_interested_pts[:, 1].tolist()]

    # statistic
    print(timer)

    # vis
    if flag_debug:
        ## draw match
        end = -1
        img_match = slam_lib.vis.draw_matches(img_left, interested_pts_2d_left_img_match[:end], img_right,
                                              interested_pts_2d_right_img_match[:end])
        cv2.namedWindow('checkboard points reprojection match', cv2.WINDOW_NORMAL)
        cv2.imshow('checkboard points reprojection match', img_match)
        cv2.waitKey(0)

        # lines
        origins = np.zeros(left_interested_lines_3d_left_cam.shape)
        lines_left = o3.geometry.LineSet()
        lines_left.points = o3.utility.Vector3dVector(np.vstack([left_interested_lines_3d_left_cam, origins]))
        lines_left.lines = o3.utility.Vector2iVector(
            np.arange(0, 2 * len(left_interested_lines_3d_left_cam)).reshape(2, -1).T)
        colors = [[1, 0, 0] for i in range(2 * len(left_interested_lines_3d_left_cam))]
        lines_left.colors = o3.utility.Vector3dVector(colors)

        lines_right = o3.geometry.LineSet()
        lines_right.points = o3.utility.Vector3dVector(np.vstack([right_interested_lines_3d_right_cam, origins]))
        lines_right.lines = o3.utility.Vector2iVector(
            np.arange(0, 2 * len(right_interested_lines_3d_right_cam)).reshape(2, -1).T)
        colors = [[200, 0, 0] for i in range(2 * len(right_interested_lines_3d_right_cam))]
        lines_right.colors = o3.utility.Vector3dVector(colors)
        lines_right.transform(np.linalg.inv(tf_left_2_right))

        # pc
        pc_general_left = o3.geometry.PointCloud()
        pc_general_left.points = o3.utility.Vector3dVector(pts_3d_in_left)

        pc_general_right = o3.geometry.PointCloud()
        pc_general_right.points = o3.utility.Vector3dVector(pts_3d_in_right)
        pc_general_right.transform(np.linalg.inv(tf_left_2_right))

        pc_interested = o3.geometry.PointCloud()
        pc_interested.points = o3.utility.Vector3dVector(left_interested_pts_3d_left_cam)

        # frames
        mesh = o3.geometry.TriangleMesh()
        frame_left = mesh.create_coordinate_frame(size=20.0)
        frame_right = mesh.create_coordinate_frame(size=20.0)
        frame_right.transform(np.linalg.inv(tf_left_2_right))

        o3.visualization.draw_geometries([pc_general_left])

        o3.visualization.draw_geometries(
            [frame_left, frame_right, lines_left, lines_right, pc_interested])

        o3.visualization.draw_geometries([pc_interested])

        pts_2d_left_img_plane_reproject, *_ = cv2.projectPoints(
            left_interested_lines_3d_right_cam_normalized, rvec=np.eye(3), tvec=np.zeros(3),
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

    return id_left_interested_pts_2_right_interested_pts


def prerecon(stereo, matcher, img_left, img_right, flag_debug=False):
    tf_left_2_right = slam_lib.mapping.rt_2_tf(stereo.r, stereo.t)

    '''extract feature from stereo'''
    # superglue
    time_start_0 = time.time()
    pts_2d_left_raw, pts_2d_right_raw = None, None

    pts_2d_left_raw, _, pts_2d_right_raw, _ = matcher.match(img_left, img_right)
    pts_2d_left_raw, pts_2d_right_raw, _ = slam_lib.geometry.epipolar_geometry_filter_matched_pts_pair(
        pts_2d_left_raw, pts_2d_right_raw)
    print('get free hair pts in', time.time() - time_start_0, 'seconds')

    # compute feature 3d coord
    pts_3d_in_left = stereo.correspondence_to_3d_in_left(pts_2d_left_raw, pts_2d_right_raw)
    pts_3d_in_right = slam_lib.mapping.transform_pt_3d(tf_left_2_right, pts_3d_in_left)

    '''vis'''
    if flag_debug:
        # match
        img3 = slam_lib.vis.draw_matches(img_left, pts_2d_left_raw, img_right, pts_2d_right_raw, flag_count_match=False)
        cv2.namedWindow('match', cv2.WINDOW_NORMAL)
        cv2.imshow('match', img3)
        cv2.waitKey(0)

        # pc
        pc_general_left = o3.geometry.PointCloud()
        pc_general_left.points = o3.utility.Vector3dVector(pts_3d_in_left)
        pc_general_left.paint_uniform_color((1, 0, 0))

        pc_general_right = o3.geometry.PointCloud()
        pc_general_right.points = o3.utility.Vector3dVector(pts_3d_in_right)
        pc_general_right.paint_uniform_color((0, 0, 1))

        o3.visualization.draw_geometries([pc_general_left, pc_general_right])

        pc_general_right.transform(np.linalg.inv(tf_left_2_right))
        o3.visualization.draw_geometries([pc_general_left, pc_general_right])

        # frames
        mesh = o3.geometry.TriangleMesh()
        frame_left = mesh.create_coordinate_frame(size=20.0)
        frame_right = mesh.create_coordinate_frame(size=20.0)
        frame_right.transform(np.linalg.inv(tf_left_2_right))
        o3.visualization.draw_geometries([pc_general_left, pc_general_right, frame_left, frame_right])

    return pts_3d_in_left, pts_3d_in_right


def match_by_2dnn(left_interested_pts_2d_left_img, left_pre_matched_pts,
                  right_interested_pts_2d_right_img, right_pre_matched_pts, distance_min=100.0):
    """

    :param left_interested_pts_2d_left_img:
    :param left_pre_matched_pts:
    :param right_interested_pts_2d_right_img:
    :param right_pre_matched_pts:
    :param distance_min:
    :return:
    """
    assert left_pre_matched_pts.shape[1] == 2
    assert left_interested_pts_2d_left_img.shape[1] == 2
    assert left_pre_matched_pts.shape == right_pre_matched_pts.shape

    'match interested pts with pre matched pts in each img'
    id_left_pre_matched_pts_2_left_interested_pts = slam_lib.geometry.nearest_neighbor_points_2_points(left_pre_matched_pts, left_interested_pts_2d_left_img, distance_min=distance_min)
    id_right_pre_matched_pts_2_right_interested_pts = slam_lib.geometry.nearest_neighbor_points_2_points(right_pre_matched_pts, right_interested_pts_2d_right_img, distance_min=distance_min)

    'match interested pts in two side img'
    id_left_pre_match_2_right_pre_match = []
    id_left_interesting_2_right_interesting = []
    id_left_interested_pts_set, id_right_interested_pts_set = set(), set()
    i_left, i_right = 0, 0
    for i in range(len(left_pre_matched_pts)):   # max possible loop number
        # if any index over range
        if i_left >= len(id_left_pre_matched_pts_2_left_interested_pts) or i_right >= len(id_right_pre_matched_pts_2_right_interested_pts):
            break

        if id_left_pre_matched_pts_2_left_interested_pts[i_left][0] > id_right_pre_matched_pts_2_right_interested_pts[i_right][0]:    # left is bigger, increase right
            i_right += 1
        elif id_left_pre_matched_pts_2_left_interested_pts[i_left][0] < id_right_pre_matched_pts_2_right_interested_pts[i_right][0]:
            i_left += 1
        else:
            if id_left_pre_matched_pts_2_left_interested_pts[i_left][1] not in id_left_interested_pts_set and \
                    id_right_pre_matched_pts_2_right_interested_pts[i_right][1] not in id_right_interested_pts_set:
                id_left_pre_match_2_right_pre_match.append([id_left_pre_matched_pts_2_left_interested_pts[i_left][0], id_right_pre_matched_pts_2_right_interested_pts[i_right][0]])
                id_left_interesting_2_right_interesting.append([id_left_pre_matched_pts_2_left_interested_pts[i_left][1], id_right_pre_matched_pts_2_right_interested_pts[i_right][1]])
                id_left_interested_pts_set.add(id_left_pre_matched_pts_2_left_interested_pts[i_left][1])
                id_right_interested_pts_set.add(id_right_pre_matched_pts_2_right_interested_pts[i_right][1])
            i_left += 1
            i_right += 1

    print(id_left_pre_match_2_right_pre_match)
    return id_left_interesting_2_right_interesting


class Matcher:
    def __init__(self, ):
        self.matcher_core = slam_lib.feature.Matcher()
        self.flag_debug = False

    def match(self, img1, pts1, img2, pts2, flag_vis=False):
        """

        :param img1:
        :param pts1:
        :param img2:
        :param pts2:
        :return:
        """

        distance_min = 20.0

        '''preprocess image'''
        if isinstance(img1, list):
            img1 = np.asarray(img1).astype(np.uint8)
        if isinstance(img2, list):
            img2 = np.asarray(img2).astype(np.uint8)
        if isinstance(pts1, list):
            pts1 = np.asarray(pts1)
        if isinstance(pts2, list):
            pts2 = np.asarray(pts2)

        print('img1 mean', np.mean(img1))
        print('img2 mean', np.mean(img2))

        print(img1.shape, img2.shape)
        print(img1.dtype, img2.dtype)

        print(pts1.shape, pts2.shape)
        print(pts1.dtype, pts2.dtype)

        # gray_left, gray_right = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        '''pre 2d match: match pts by superglue'''
        print('getting superglue match pts')
        pts_2d_left_super, _, pts_2d_right_super, _ = self.matcher_core.match(img1, img2)
        pts_2d_left_sift, _, pts_2d_right_sift, _ = \
            slam_lib.feature.get_epipolar_geometry_filtered_sift_matched_pts(img1, img2, shrink=0.8)

        # pts_2d_left_super, pts_2d_right_super, _ = slam_lib.geometry.epipolar_geometry_filter_matched_pts_pair(pts_2d_left_super, pts_2d_right_super)

        pts_2d_left_pre, pts_2d_right_pre = np.concatenate([pts_2d_left_super, pts_2d_left_sift], axis=0),\
                                            np.concatenate([pts_2d_right_super, pts_2d_right_sift], axis=0)

        print('got superglue matched pts, # of pt : ', len(pts_2d_left_super), pts_2d_right_super.shape)
        print('got sift matched pts, # of pt : ', len(pts_2d_left_sift), pts_2d_left_sift.shape)
        print('Check shape', pts_2d_left_pre.shape, pts_2d_right_pre.shape)

        '''with the help of super point match: match left and right interesting pts'''
        if len(pts_2d_right_pre) == 0 and len(pts_2d_right_pre) == 0:
            print('could\'t get pre match pts')
            return []
        else:
            print('got pre match ')

        id_left_interested_2_right_interested = match_by_2dnn(pts1, pts_2d_left_pre,
                                                              pts2, pts_2d_right_pre, distance_min=distance_min)
        print('got match for interesting pts, # of match: ', len(id_left_interested_2_right_interested))

        if flag_vis:
            # img_match_super = slam_lib.vis.draw_matches(img1, pts_2d_left_super, img2, pts_2d_right_super)
            # cv2.namedWindow('super match', cv2.WINDOW_NORMAL)
            # cv2.imshow('super match', img_match_super)
            # cv2.waitKey(0)
            self.draw_matches(img1, pts_2d_left_super, img2, pts_2d_right_super)

            # img_match_sift = slam_lib.vis.draw_matches(img1, pts_2d_left_sift, img2, pts_2d_right_sift)
            # cv2.namedWindow('sift match', cv2.WINDOW_NORMAL)
            # cv2.imshow('sift match', img_match_super)
            # cv2.waitKey(0)
            self.draw_matches(img1, pts_2d_left_sift, img2, pts_2d_right_sift)

            self.draw_matches(img1, pts1, img2, pts2, id_left_interested_2_right_interested)
        return id_left_interested_2_right_interested

    def draw_matches(self, img1, pts1, img2, pts2, match_id=None):
        """

        :param img1:
        :param pts1:
        :param img2:
        :param pts2:
        :param match_id:
        :return:
        """
        if match_id is None:
            match_id = []
        '''preprocess input'''
        if len(match_id) == 0:
            if len(pts1) == len(pts2):
                match_id = np.concatenate([np.arange(0, len(pts1))[None, :], np.arange(0, len(pts1))[None, :]], axis=0)
            else:
                return
        if isinstance(img1, list):
            img1 = np.asarray(img1).astype(np.uint8)
        if isinstance(img2, list):
            img2 = np.asarray(img2).astype(np.uint8)
        if isinstance(pts1, list):
            pts1 = np.asarray(pts1)
        if isinstance(pts2, list):
            pts2 = np.asarray(pts2)
        if isinstance(match_id, list):
            match_id = np.asarray(match_id)

        pts1 = pts1[match_id[:, 0]]
        pts2 = pts2[match_id[:, 1]]

        for pt1, pt2 in zip(pts1, pts2):
            pt1, pt2 = pt1.reshape(-1, 2), pt2.reshape(-1, 2)
            img_match = slam_lib.vis.draw_matches(img1, pt1, img2, pt2)
            cv2.namedWindow('match', cv2.WINDOW_NORMAL)
            cv2.imshow('match', img_match)
            key_pressed = cv2.waitKey(0)
            if key_pressed == 'e':
                break
        img_match = slam_lib.vis.draw_matches(img1, pts1, img2, pts2)
        cv2.namedWindow('match after all', cv2.WINDOW_NORMAL)
        cv2.imshow('match after all', img_match)
        cv2.waitKey(0)


def main():
    matcher = Matcher()

    img_last = None
    img_dir = '/home/cheng/proj/data/AutoPano/data/graf/'
    for i, img_name in enumerate(os.listdir(img_dir)):
        if i == 0:
            img_last = img_dir + img_name
            continue
        img_left, img_right = cv2.imread(img_last), cv2.imread(img_dir + img_name)

        '''get interested pts'''
        print('getting interesting pts')
        left_interested_pts_2d_left_img, des1, right_interested_pts_2d_right_img, des2 = \
            slam_lib.feature.get_sift_pts_and_sift_feats(img_left, img_right, shrink=1.0, flag_debug=True)

        print('got interesting pts', len(left_interested_pts_2d_left_img), 'pts from left',
              len(right_interested_pts_2d_right_img), 'pts from right')

        '''match interesting pts'''
        id_left_interested_2_right_interested = matcher.match(img_left, left_interested_pts_2d_left_img,
                                                              img_right, right_interested_pts_2d_right_img)
        img_last = img_dir + img_name

        matcher.draw_matches(img_left, left_interested_pts_2d_left_img,
                             img_right, right_interested_pts_2d_right_img, id_left_interested_2_right_interested)


if __name__ == '__main__':
    main()
