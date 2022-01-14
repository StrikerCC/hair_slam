import cv2
import numpy as np
import open3d as o3


def main():
    camera = StereoCamera()
    src_l, src_r = cv2.cv2.imread('/home/cheng/proj/3d/BiCameraSDKv2.0/test_data/calibration_data/left/02.bmp'), \
                   cv2.cv2.imread('/home/cheng/proj/3d/BiCameraSDKv2.0/test_data/calibration_data/right/02.bmp')

    # src_l, src_r = cv2.cv2.imread('/home/cheng/proj/3d/BiCameraSDKv2.0/data/LeftCamera/005.bmp'), \
    #                cv2.cv2.imread('/home/cheng/proj/3d/BiCameraSDKv2.0/data/RightCamera/005.bmp')

    imagesize = src_l.shape[:-1]
    imagesize_rect = (int(src_l.shape[0] * 1.8), int(src_l.shape[1] * 1.2))

    cv2.cv2.namedWindow('left', cv2.cv2.WINDOW_NORMAL)
    cv2.cv2.namedWindow('right', cv2.cv2.WINDOW_NORMAL)
    cv2.cv2.imshow('left', src_l)
    cv2.cv2.imshow('right', src_r)
    cv2.cv2.waitKey(0)
    cv2.cv2.destroyAllWindows()

    '''undistorted image'''
    undest_l = cv2.cv2.undistort(src_l, camera.leftCameraMatrix, distCoeffs=camera.leftDistCoeffs)
    undest_r = cv2.cv2.undistort(src_r, camera.rightCameraMatrix, distCoeffs=camera.rightDistCoeffs)

    cv2.cv2.namedWindow('left', cv2.cv2.WINDOW_NORMAL)
    cv2.cv2.namedWindow('right', cv2.cv2.WINDOW_NORMAL)
    cv2.cv2.imshow('left', undest_l)
    cv2.cv2.imshow('right', undest_r)
    cv2.cv2.waitKey(0)
    cv2.cv2.destroyAllWindows()

    '''rectify image'''
    R_l, R_r, P_l, P_r, *_ = cv2.cv2.stereoRectify(camera.leftCameraMatrix, camera.leftDistCoeffs, camera.rightCameraMatrix, camera.rightDistCoeffs, imagesize_rect, camera.R, camera.T)
    rect_map_x_l, rect_map_y_l = cv2.cv2.initUndistortRectifyMap(camera.leftCameraMatrix, camera.leftDistCoeffs, R_l, P_l, imagesize_rect, cv2.cv2.CV_32FC1)
    rect_map_x_r, rect_map_y_r = cv2.cv2.initUndistortRectifyMap(camera.rightCameraMatrix, camera.rightDistCoeffs, R_r, P_r, imagesize_rect, cv2.cv2.CV_32FC1)

    '''apply mappping'''
    img_l = cv2.cv2.remap(src_l, rect_map_x_l, rect_map_y_l, cv2.cv2.INTER_LINEAR)
    img_r = cv2.cv2.remap(src_r, rect_map_x_r, rect_map_y_r, cv2.cv2.INTER_LINEAR)

    cv2.cv2.namedWindow('left', cv2.cv2.WINDOW_NORMAL)
    cv2.cv2.namedWindow('right', cv2.cv2.WINDOW_NORMAL)
    cv2.cv2.imshow('left', img_l)
    cv2.cv2.imshow('right', img_r)
    cv2.cv2.waitKey(0)
    cv2.cv2.destroyAllWindows()

    '''extract feature'''
    feat_detector = cv2.cv2.SIFT_create()
    keypoints_l, feat_l = feat_detector.detectAndCompute(img_l, None)
    keypoints_r, feat_r = feat_detector.detectAndCompute(img_r, None)

    # drawing the key points on the input image using drawKeypoints() function
    resultimage_l = cv2.cv2.drawKeypoints(img_l, keypoints_l, 0, (0, 255, 0), flags=cv2.cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    resultimage_r = cv2.cv2.drawKeypoints(img_r, keypoints_r, 0, (0, 255, 0), flags=cv2.cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # displaying the image with keypoints as the output on the screen
    cv2.cv2.namedWindow('left_image_with_keypoints', cv2.cv2.WINDOW_NORMAL)
    cv2.cv2.namedWindow('right_image_with_keypoints', cv2.cv2.WINDOW_NORMAL)
    cv2.cv2.imshow('left_image_with_keypoints', resultimage_l)
    cv2.cv2.imshow('right_image_with_keypoints', resultimage_r)
    cv2.cv2.waitKey(0)

    # BFMatcher解决匹配
    bf = cv2.cv2.BFMatcher()
    matches = bf.knnMatch(feat_l, feat_r, k=2)

    '''filter match'''
    good = []
    better = []
    pts_l = []
    pts_r = []
    for m, n in matches:
        if m.distance < 1.75 * n.distance:
            good.append([m])
            pts_l.append(keypoints_l[m.queryIdx].pt)
            pts_r.append(keypoints_r[n.trainIdx].pt)

    pts_l = np.int32(pts_l)
    pts_r = np.int32(pts_r)
    F, mask = cv2.cv2.findFundamentalMat(pts_l, pts_r, cv2.cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99,
                                         maxIters=1000)
    mask_ = mask.ravel() == 1
    for i in range(len(mask_)):
        if mask_[i] == 1:
            better.append(good[i])
    pts_l = pts_l[mask_]
    pts_r = pts_r[mask_]

    pts_l_homo = np.concatenate([pts_l, np.ones((pts_l.shape[0], 1))], axis=-1)
    pts_r_homo = np.concatenate([pts_r, np.ones((pts_r.shape[0], 1))], axis=-1)
    error = np.trace(np.matmul(pts_r_homo.T, np.matmul(pts_l_homo, F))) / pts_r_homo.shape[0]
    print("Before filter", len(good))
    print("After filter", len(better))
    print(error)

    img4 = cv2.cv2.drawMatchesKnn(img_l, keypoints_l, img_r, keypoints_r, good, None, flags=2)
    img5 = cv2.cv2.drawMatchesKnn(img_l, keypoints_l, img_r, keypoints_r, better, None, flags=2)

    cv2.cv2.namedWindow("BFmatch", cv2.cv2.WINDOW_NORMAL)
    cv2.cv2.imshow("BFmatch", img4)
    cv2.cv2.namedWindow("BFmatchFiltered", cv2.cv2.WINDOW_NORMAL)
    cv2.cv2.imshow("BFmatchFiltered", img5)
    cv2.cv2.waitKey(0)
    cv2.cv2.destroyAllWindows()

    '''compute depth'''
    pts = []
    for pt_l, pt_r in zip(pts_l, pts_r):
        if abs(pt_l[1] - pt_r[1]) > 10.0:
            print(pt_l, pt_r)
            continue
        else:
            print('     ', pt_l, pt_r)
        pt = camera.transform_rectify_pixel_to_world_coordiante(pt_l, pt_r)
        pts.append(pt)
        print('     ', pt)

    print('Final ', len(pts))
    print(pts)
    '''show the depth'''
    objs = []
    mesh = o3.geometry.TriangleMesh()
    origin = mesh.create_coordinate_frame(size=10.0)
    objs.append(origin)

    # for pt in pts:
    #     org = mesh.create_coordinate_frame(size=10.0, origin=pt)
    #     objs.append(org)

    pts = np.asarray(pts)
    pc = o3.geometry.PointCloud()
    pc.points = o3.utility.Vector3dVector(pts)
    objs.append(pc)
    o3.visualization.draw_geometries(objs)

    return


if __name__ == '__main__':
    main()