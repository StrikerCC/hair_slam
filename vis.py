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


def main():
    img, info = read.format_data()
    draw_lines(img, info)


if __name__ == '__main__':
    main()
