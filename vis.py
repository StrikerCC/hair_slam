# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 9/2/21 4:39 PM
"""
import cv2
import numpy as np

import read


def draw_lines(img, info):
    # TODO:
    for hair_info in info:
        start, end = hair_info['follicle'], hair_info['hair_end']
        cv2.line(img, end, np.asarray(end)+1, color=(0, 0, 155), thickness=3)
    cv2.imshow('hairs', img)
    cv2.waitKey(0)


def main():
    img, info = read.format_data()
    draw_lines(img, info)


if __name__ == '__main__':
    main()
