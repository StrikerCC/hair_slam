# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 9/2/21 4:39 PM
"""
import cv2
import numpy as np

import read


def draw_lines(img, info, name):
    # TODO:
    for hair_info in info:
        start, end = hair_info['follicle'], hair_info['hair_end']
        cv2.line(img, start, np.asarray(start)+1, color=(255, 0, 0), thickness=3)   # root
        cv2.line(img, end, np.asarray(start), color=(0, 255, 0), thickness=1)       # hair
        cv2.line(img, end, np.asarray(end)+1, color=(0, 0, 255), thickness=3)       # head
    cv2.imshow(name+'hairs', img)
    cv2.waitKey(0)


def draw_matches(img1, pts1, img2, pts2):
    if len(img1.shape) < 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) < 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    if pts1.shape[1] != 2:
        pts1 = pts1.reshape(-1, 2)
    if pts2.shape[1] != 2:
        pts2 = pts2.reshape(-1, 2)

    pts1, pts2 = pts1.astype(int), pts2.astype(int)
    img = np.copy(np.concatenate([img1, img2], axis=1))
    pts2[:, 0] = pts2[:, 0] + img1.shape[1]
    for i in range(len(pts1)):
        pt1, pt2 = pts1[i], pts2[i]
        color = np.random.random(3) * 255
        img = cv2.circle(img, pt1, radius=8, color=color, thickness=3)
        img = cv2.circle(img, pt2, radius=8, color=color, thickness=3)
        img = cv2.line(img, pt1, pt2, color=color, thickness=2)
    return img


def main():
    img, info = read.format_data()
    draw_lines(img, info)


if __name__ == '__main__':
    main()
