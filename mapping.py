# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 11/16/21 10:09 AM
"""
import random
from cam import StereoCamera
import cv2.cv2
import numpy as np
import open3d as o3
import transforms3d as tf3


def transform_pt_3d(tf, pts):
    assert tf.shape == (4, 4)
    if pts.shape[0] == 3:
        return np.matmul(tf[:3, :3], pts) + tf[:3, -1:]
    elif pts.shape[1] == 3:
        return np.matmul(pts, tf[:3, :3].T) + tf[:3, -1:].T
    else:
        raise ValueError('input points shape invalid, expect (n, 3) or (3, n), but get ' + str(pts.shape))


def umeyama_ransac(src, tgt, max_iter=100, confidence=0.9, max_error=0.5):  # compute tf by common pts
    if src.shape[1] != 3:
        src = src.T
    if tgt.shape[1] != 3:
        tgt = tgt.T
    assert src.shape[1] == 3 and tgt.shape[1] == 3, str(src.shape) + ', ' + str(src.shape)

    num_match_pts = 4

    max_inlier_ratio = 0.0
    tf_best = None

    index_list = np.arange(0, len(src))
    for iter in range(max_iter):
        index = np.random.choice(index_list, num_match_pts, replace=False).tolist()  # random pick

        # compute tf
        tf = umeyama(src=src[index].T, tgt=tgt[index].T)

        # compute inlier ratio
        error = np.linalg.norm(tgt - transform_pt_3d(tf, src), axis=-1)      # Manhattan distance
        inlier_ratio = np.count_nonzero(error < max_error) / len(src)

        # break if needed
        if inlier_ratio > confidence:
            return tf

        # update max
        if inlier_ratio >= max_inlier_ratio:
            max_inlier_ratio = inlier_ratio
            tf_best = tf

        print(index)
        print(inlier_ratio)
        print(error)
        print(tf3.euler.mat2euler(tf[:3, :3]))
        print(tf[:3, -1])
        print()

    return tf_best


def umeyama(src, tgt):
    if src.shape[0] != 3:
        src = src.T
    if tgt.shape[0] != 3:
        tgt = tgt.T
    assert src.shape[0] == 3 and tgt.shape[0] == 3, str(src.shape) + ', ' + str(src.shape)

    tf = np.eye(len(src) + 1)
    # rotation
    src_tgt_cov = np.matmul(tgt-np.expand_dims(np.mean(tgt, axis=1), axis=1), np.transpose(src-np.expand_dims(np.mean(src, axis=1), axis=1)))
    u, lam, vt = np.linalg.svd(src_tgt_cov)
    s = np.eye(len(src))
    if np.linalg.det(u) * np.linalg.det(vt) < -0.5:
        s[-1, -1] = -1.0
    tf[:len(src), :len(src)] = np.matmul(np.matmul(u, s), vt)

    # translation
    tf[:-1, -1] = np.mean(tgt, axis=1) - np.mean(np.matmul(tf[:len(src), :len(src)], src), axis=1)
    return tf


def main():
    pts = np.random.random((30, 3))
    # pts = np.vstack([pts, pts+1, pts+2])

    angle_gt = (1, 1, 1)
    tf_gt = np.eye(4)
    tf_gt[:3, :3] = tf3.euler.euler2mat(*angle_gt)
    tf_gt[:3, -1] = (10, 0, 0)

    # print(pts)

    # tf
    pts_tgt = transform_pt_3d(tf_gt, pts)

    # compute tf
    tf = umeyama_ransac(src=pts, tgt=pts_tgt)
    angle = tf3.euler.mat2euler(tf[:3, :3])

    print(pts_tgt)
    print(transform_pt_3d(tf, pts))
    print(np.allclose(pts_tgt, transform_pt_3d(tf, pts)))
    print(angle_gt)
    print(tf_gt[:3, -1])
    print(angle)
    print(tf[:3, -1])

    return


if __name__ == '__main__':
    main()

