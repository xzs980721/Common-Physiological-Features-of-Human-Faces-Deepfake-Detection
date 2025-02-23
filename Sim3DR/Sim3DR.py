# coding: utf-8

import os
import sys
import numpy as np
import Sim3DR_Cython

# 添加路径
def _init_paths():
    this_dir = os.path.dirname(__file__)
    lib_path = os.path.join(this_dir, '..', 'lib')
    sys.path.insert(0, lib_path)

# 确保在导入其他模块之前调用 _init_paths
_init_paths()

def get_normal(vertices, triangles):
    normal = np.zeros_like(vertices, dtype=np.float32)
    Sim3DR_Cython.get_normal(normal, vertices, triangles, vertices.shape[0], triangles.shape[0])
    return normal

def rasterize(vertices, triangles, colors, bg=None,
              height=None, width=None, channel=None,
              reverse=False):
    if bg is not None:
        height, width, channel = bg.shape
    else:
        assert height is not None and width is not None and channel is not None
        bg = np.zeros((height, width, channel), dtype=np.uint8)

    buffer = np.zeros((height, width), dtype=np.float32) - 1e8

    if colors.dtype != np.float32:
        colors = colors.astype(np.float32)
    Sim3DR_Cython.rasterize(bg, vertices, triangles, colors, buffer, triangles.shape[0], height, width, channel,
                            reverse=reverse)
    return bg