# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/30/21 2:56 PM
"""
import os
import cv2
import numpy as np


class PinHoleCamera:
    def __init__(self):
        self.camera_matrix = None
        self.pose = None
        self.distortion_coefficient = None

    def proj(self, pts_3d):
        """
        project 3d points in world frame to pixel frame
        :param pts_3d: (n, 3)
        :type pts_3d:
        :return:
        :rtype:
        """
        if pts_3d.shape[0] != 3:
            pts_3d = pts_3d.T   # (3, N)
        assert len(pts_3d.shape) == 2 and pts_3d.shape[0] == 3

        pts_3d_homo = np.vstack(pts_3d, np.ones(pts_3d.shape[1]))   # (4, N)
        pts_2d_line = np.matmul(self.camera_matrix, np.matmul(np.linalg.inv(self.pose), pts_3d_homo)[:3, :])    # (3, N)
        pts_2d = pts_2d_line[:2, :] / pts_2d_line[-1, :]    # (2, N)
        return pts_2d.T     # (2, N)


class BiCamera:

    # leftCameraMatrix, rightCameraMatrix = None, None
    # leftDistCoeffs, rightDistCoeffs = None, None
    # R, T, E, F = None, None, None, None
    # R1, P1, R2, P2, Q = None, None, None, None, None

    def __init__(self, para_file_path=None):
        self.cam_left, self.cam_right = PinHoleCamera(), PinHoleCamera()
        if para_file_path is not None or os.path.isfile(str(para_file_path)):
            f = cv2.cv2.FileStorage(para_file_path, cv2.cv2.FILE_STORAGE_READ)
            self.cam_left.camera_matrix, self.cam_right.camera_matrix = f.getNode('leftCameraMatrix').mat(), f.getNode('rightCameraMatrix').mat()
            self.cam_left.distortion_coefficient, self.cam_right.distortion_coefficient = f.getNode('leftDistCoeffs').mat(), f.getNode('rightDistCoeffs').mat()
            self.rotation, self.translation, self.essential_matrix, self.fundamental_matrix = f.getNode('R').mat(), f.getNode('T').mat(), f.getNode('E').mat(), f.getNode('F').mat()
            self.R1, self.P1, self.R2, self.P2, self.Q = f.getNode('R1').mat(), f.getNode('P1').mat(), f.getNode('R2').mat(), f.getNode('P2').mat(), f.getNode('Q').mat()

    def transform_pixel_to_world_coordiante(self, leftPoint, rightPoint):
        tmpLeftPoint, tmpRightPoint = leftPoint, rightPoint
        # tmpLeftPoint.append(leftPoint)
        # tmpLeftPoint = tmpLeftPoint.reshape((1, 1, 2))
        # tmpLeftPoint = np.array([[[2000.0, 1.0],]])
        # cv2.cv2.undistortPoints(tmpLeftPoint, self.leftCameraMatrix, self.leftDistCoeffs, R=self.R1, P=self.P1)
        # tmpRightPoint.append(rightPoint)

        # tmpRightPoint = tmpRightPoint.reshape((1, 1, 2))
        # tmpRightPoint = np.array([[[-100.0, 1.0],]])
        # cv2.cv2.undistortPoints(tmpRightPoint, self.rightCameraMatrix, self.rightDistCoeffs, R=self.R2, P=self.P2)

        # leftPoint = tmpLeftPoint[0]
        # rightPoint = tmpRightPoint[0]

        disparity = leftPoint[0] - rightPoint[0]
        _Q = self.Q
        homg_pt = np.matmul(_Q, np.array([leftPoint[0], leftPoint[1], disparity, 1.0]))
        realPoint = homg_pt[:-1]
        realPoint /= homg_pt[3]

        worldPoint = np.zeros(3)
        worldPoint[0] = realPoint[0]
        worldPoint[1] = realPoint[1]
        worldPoint[2] = realPoint[2]
        return worldPoint


if __name__ == '__main__':
    BiCamera = BiCamera('/home/cheng/proj/3d/BiCameraSDKv2.0/test_data/saved_parameters.xml')
    pt_left, pt_right = np.array([220, 220]), np.array([100, 100])
    world = BiCamera.transform_pixel_to_world_coordiante(pt_left, pt_right)
    print(world)
