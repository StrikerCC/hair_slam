# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/30/21 3:17 PM
"""
import numpy as np


class Frame:
    def __init__(self):
        self.id = None
        self.img = None
        self.pose = None
        self.pts_2d = None
        self.pts_3d = None
        self.mapping = None

