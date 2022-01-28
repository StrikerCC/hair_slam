# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import copy
import cv2.cv2
import numpy as np
import matplotlib.pyplot as plt
import slam_lib.read
import slam_lib.vis
from slam_lib.read import readtxt
import open3d as o3


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, f = img1.shape
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def computeF(img_l, img_r):
    siftfeatures = cv2.cv2.xfeatures2d.SIFT_create()
    # siftfeatures = cv2.cv2.ORB_create()

    '''feature'''
    keypoints_l, feat_l = siftfeatures.detectAndCompute(img_l, None)
    keypoints_r, feat_r = siftfeatures.detectAndCompute(img_r, None)

    # drawing the key points on the input image using drawKeypoints() function
    resultimage_l = cv2.drawKeypoints(img_l, keypoints_l, 0, (0, 255, 0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    resultimage_r = cv2.drawKeypoints(img_r, keypoints_r, 0, (0, 255, 0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # displaying the image with keypoints as the output on the screen
    cv2.imshow('left_image_with_keypoints', resultimage_l)
    cv2.imshow('right_image_with_keypoints', resultimage_r)
    cv2.waitKey(0)

    # BFMatcher解决匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(feat_l, feat_r, k=2)

    # 调整ratio
    good = []
    pts_l = []
    pts_r = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            pts_l.append(keypoints_l[m.queryIdx].pt)
            pts_r.append(keypoints_r[n.trainIdx].pt)

    img5 = cv2.drawMatchesKnn(img_l, keypoints_l, img_r, keypoints_r, good, None, flags=2)

    cv2.cv2.namedWindow("BFmatch", cv2.cv2.WINDOW_NORMAL)
    cv2.imshow("BFmatch", img5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pts_l = np.int32(pts_l)
    pts_r = np.int32(pts_r)
    F, mask = cv2.cv2.findFundamentalMat(pts_l, pts_r, cv2.cv2.FM_LMEDS, ransacReprojThreshold=5, confidence=0.8,
                                         maxIters=1000)

    # We select only inlier points
    # pts_l = pts_l[mask.ravel() == 1]
    # pts_r = pts_r[mask.ravel() == 1]
    #
    # lines_l = cv2.cv2.computeCorrespondEpilines(pts_r.reshape(-1, 1, 2), 2, F)
    # lines_l = lines_l.reshape(-1, 3)
    # lines_r = cv2.cv2.computeCorrespondEpilines(pts_l.reshape(-1, 1, 2), 2, F)
    # lines_r = lines_r.reshape(-1, 3)
    #
    # img_5, img_6 = drawlines(img_l, img_r, lines_l, pts_r, pts_l)
    # img_3, img_4 = drawlines(img_r, img_l, lines_r, pts_r, pts_l)
    #
    # plt.subplot(121), plt.imshow(img_5)
    # plt.subplot(122), plt.imshow(img_3)
    # plt.show()

    print(F)
    return F


def main():
    F = np.array([[3.13718710e-06, -1.52784098e-05, 1.01005367e-02],
                  [8.22328436e-06, 2.91938491e-06, 8.82566248e-03],
                  [-1.20338483e-02, 1.02307245e-02, 1.00000000e+00]])

    # img_l = cv2.cv2.imread('/home/cheng/proj/3d/hair_host/bin/left.jpg')
    # img_r = cv2.cv2.imread('/home/cheng/proj/3d/hair_host/bin/right.jpg')

    img_l, data_l = slam_lib.read.format_data(img_path='/home/cheng/proj/3d/hair_host/bin/left.jpg',
                                     data_json_path='/home/cheng/proj/3d/hair_host/bin/left_hair_info.json')
    img_l_copy = copy.deepcopy(img_l)
    slam_lib.vis.draw_lines(img_l_copy, data_l, 'l')

    img_r, data_r = slam_lib.read.format_data(img_path='/home/cheng/proj/3d/hair_host/bin/right.jpg',
                                     data_json_path='/home/cheng/proj/3d/hair_host/bin/right_hair_info.json')
    img_r_copy = copy.deepcopy(img_r)
    slam_lib.vis.draw_lines(img_r_copy, data_r, 'r')
    # F = computeF(img_l, img_r)

    '''take key points to np array'''
    pts_l = [np.asarray(data['follicle']) for data in data_l]
    pts_r = [np.asarray(data['follicle']) for data in data_r]
    # pts_l, pts_r = np.asarray(pts_l), np.asarray(pts_r)

    # print(pts_r.shape)

    '''compute left frame key points coord in right camera frame'''
    # get homogenous coord by solving the epipline equation
    # for pt_l, pt_r in pts_l, pts_r:

    # reprojected to right frame

    return


def compute_rigid_transf(pc_target, pc_src):
    dimension = 3
    assert isinstance(pc_target, np.ndarray) and isinstance(pc_src, np.ndarray)
    assert pc_target.shape == pc_src.shape
    # assert pc_target.shape[1] == pc_src.shape[1] == dimension
    # compute H
    H = np.matmul(pc_src, pc_target.T)
    U, S, V_T = np.linalg.svd(H)
    R = np.matmul(V_T.T, U)
    t = pc_target.mean(axis=-1) - np.matmul(R, pc_src).mean(axis=-1)

    # assert np.linalg.det(R) == 1
    return R, t


def main_():
    cam, ct = readtxt()
    pt_cam, pt_ct = np.array(cam), np.array(ct)

    pc_cam = o3.geometry.PointCloud()
    pc_ct = o3.geometry.PointCloud()

    pc_cam.points = o3.utility.Vector3dVector(np.array(cam))
    pc_cam.paint_uniform_color((1, 0, 0))

    pc_ct.points = o3.utility.Vector3dVector(np.array(ct))
    pc_ct.paint_uniform_color((0, 0, 1))

    # result = o3.pipelines.registration.registration_icp(source=pc_cam, target=pc_ct, max_correspondence_distance=1.0,
    #                                                     init=np.eye(4), estimation_method=o3.pipelines.registration.TransformationEstimationPointToPoint(),)
    transformation = np.eye(4)
    R, t = compute_rigid_transf(pc_target=np.array(ct).T, pc_src=np.array(cam).T)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    pc_cam = pc_cam.transform(transformation)
    pt_cam = np.asarray(pc_cam.points)
    print(transformation)

    # mesh = o3.geometry.TriangleMesh()
    #
    # sphere_cam = []
    # for pt in cam:
    #     sphere_cam.append(mesh.create_coordinate_frame(size=20.0, origin=pt))
    #
    # sphere_ct = []
    # for pt in ct:
    #     sphere_ct.append(mesh.create_sphere(radius=5.0, resolution=5))
    #     tsfm = np.eye(4)
    #     tsfm[0, 3] = pt[0]
    #     tsfm[1, 3] = pt[1]
    #     tsfm[2, 3] = pt[2]
    #     sphere_ct[-1].transform(tsfm)
    #     # o3.visualization.draw_geometries(sphere_ct)
    # o3.visualization.draw_geometries(sphere_ct + sphere_cam)
    #
    # for sphere in sphere_cam:
    #     sphere.transform(transformation)
    # o3.visualization.draw_geometries(sphere_ct + sphere_cam)

    print(pt_cam)
    print(pt_ct)
    print(np.linalg.norm(pt_cam-pt_ct) / 6)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
