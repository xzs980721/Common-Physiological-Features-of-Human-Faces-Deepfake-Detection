import os
import scipy.io as sio
import os.path as osp
import torch
import numpy as np
from torchvision import datasets, transforms
from Sim3DR import rasterize
from utils.functions import plot_image
from utils.io import _load
import yaml
from utils.tddfa_util import _to_ctype, TDDFA
import cv2
from functools import lru_cache
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import sys

import matplotlib.pyplot as plt
import time


import torch.nn as nn
@lru_cache(maxsize=1)  # 只缓存最近一次加载的结果
class DDFA():

# 设置日志配置

    def load_uv_coords(fp):
        try:
            abs_path = os.path.abspath(fp)
            logging.info(f"Loading UV coordinates from: {abs_path}")
            C = sio.loadmat(abs_path)
            uv_coords = C['UV'].copy(order='C').astype(np.float32)
            return uv_coords
        except FileNotFoundError:
            logging.error(f"File not found: {abs_path}")
            raise
        except Exception as e:
            logging.error(f"Failed to load UV coordinates from {abs_path}: {e}")
            raise

    def process_uv(uv_coords, uv_h=256, uv_w=256):
        uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
        uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
        uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
        uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1), dtype=np.float32)))  # add z
        return uv_coords

    def get_colors(img, ver):
        # nearest-neighbor sampling
        [h, w, _] = img.shape
        ver[0, :] = np.minimum(np.maximum(ver[0, :], 0), w - 1)  # x
        ver[1, :] = np.minimum(np.maximum(ver[1, :], 0), h - 1)  # y
        ind = np.round(ver).astype(np.int32)
        colors = img[ind[1, :], ind[0, :], :]  # n x 3

        return colors

    def bilinear_interpolate(img, x, y):
        """
        https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
        """
        x0 = np.floor(x).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(y).astype(np.int32)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, img.shape[1] - 1)
        x1 = np.clip(x1, 0, img.shape[1] - 1)
        y0 = np.clip(y0, 0, img.shape[0] - 1)
        y1 = np.clip(y1, 0, img.shape[0] - 1)

        i_a = img[y0, x0]
        i_b = img[y1, x0]
        i_c = img[y0, x1]
        i_d = img[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        return wa[..., np.newaxis] * i_a + wb[..., np.newaxis] * i_b + wc[..., np.newaxis] * i_c + wd[..., np.newaxis] * i_d

   

    def uv_tex(img, ver_lst, tri, g_uv_coords, uv_h=299, uv_w=299, uv_c=3, show_flag=False, wfp=None):
        uv_coords = DDFA.process_uv(g_uv_coords, uv_h=uv_h, uv_w=uv_w)
        res_lst = []
        for ver_ in ver_lst:
            ver = _to_ctype(ver_.T)  # transpose to m x 3
            colors = DDFA.bilinear_interpolate(img, ver[:, 0], ver[:, 1]) / 255.
            # `rasterize` here serves as texture sampling, may need to optimization
            res = rasterize(uv_coords, tri, colors, height=uv_h, width=uv_w, channel=uv_c)
            res_lst.append(res)

        # concat if there more than one image
        res = np.concatenate(res_lst, axis=1) if len(res_lst) > 1 else res_lst[0]

        if wfp is not None:
            cv2.imwrite(wfp, res)
            print(f'Save visualization result to {wfp}')

        if show_flag:
            plot_image(res)

        # 假设 uv_h 和 uv_w 分别是 res 的高度和宽度
        uv_h, uv_w = res.shape[:2]

        # 创建一个与原始图像大小相同的黑色掩码
        mask1 = np.zeros_like(res)

        # 截取左右眼部分
        left_eye_region = res[uv_h//4:uv_h//3, uv_w//4:uv_w//3]
        right_eye_region = res[uv_h//4:uv_h//3, 2*uv_w//3:3*uv_w//4]

        # 计算左右眼的边界
        left_eye_left = uv_w // 4
        left_eye_right = uv_w // 3
        left_eye_top = uv_h // 4
        left_eye_bottom = uv_h // 3

        right_eye_left = 2 * uv_w // 3
        right_eye_right = 3 * uv_w // 4
        right_eye_top = uv_h // 4
        right_eye_bottom = uv_h // 3

        # 计算最小包含矩形的边界
        min_left = min(left_eye_left, right_eye_left)
        min_right = max(left_eye_right, right_eye_right)
        min_top = min(left_eye_top, right_eye_top)
        min_bottom = max(left_eye_bottom, right_eye_bottom)
        offset = 10  # 可以根据需要调整偏移量

        # 确保边界不超出图像范围
        min_left = max(0, min_left - offset)
        min_right = min(uv_w, min_right + offset)
        min_top = max(0, min_top - offset)
        min_bottom = min(uv_h, min_bottom + offset)

        # 将眼睛区域复制到掩码中
        mask1[min_top:min_bottom, min_left:min_right] = res[min_top:min_bottom, min_left:min_right]

        # # # 保存处理后的图像
        # output_dir = "C:\Users\Robin\Desktop\111"
        # os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
        # output_path = os.path.join(output_dir, 'eyes_masked.png')
        # cv2.imwrite(output_path, mask1)
        # print(f'Saved eyes masked image to {output_path}')

                                # 截取鼻子部分
        nose_region = res[uv_h//4:uv_h//2, uv_w//3:2*uv_w//3]

         # 创建一个与原始图像大小相同的黑色掩码
        mask = np.zeros_like(res)

         # 将鼻子部分的像素值复制到掩码中
        mask[uv_h//4:uv_h//2, uv_w//3:2*uv_w//3] = nose_region

        #             #保存处理后的图像
        # nose_region_path = r"C:\Users\Robin\Desktop\111\nose_region_masked.png"
        # cv2.imwrite(nose_region_path, mask)
        # print(f'Saved nose region masked image to {nose_region_path}')
                # 显示眼睛区域的图像

        return mask,mask1

    def load_configs():
        config_path = r'Video-Transformer-for-Deepfake-Detection-main/configs/mb1_120x120.yml'
        cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)
        # Init FaceBoxes and TDDFA, recommend using onnx flag
        onnx_flag = True# or True to use ONNX to speed up
        if onnx_flag:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'

            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from TDDFA_ONNX import TDDFA_ONNX
            from FaceBoxes import FaceBoxes

            face_boxes = FaceBoxes_ONNX()
            tddfa = TDDFA_ONNX(**cfg)
        else:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'
            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from TDDFA_ONNX import TDDFA_ONNX
            from FaceBoxes import FaceBoxes
            tddfa = TDDFA(gpu_mode=True, **cfg)
            face_boxes = FaceBoxes()
        return tddfa, face_boxes
    
    def generate_uv_tex(a,c, features, sequence_embedding, linear_1, linear_3, proj, seq_embed, hybrid, variant, device):
        to_pil_image = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        to_normalized = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if variant == 'image':
            all_concated_feats = torch.zeros(1, 324, 768).to(device)
        else:
            b = 1
            all_concated_feats = torch.zeros(1, 64, 768).to(device)
        for i in range(b):
            uv_tensor = a.permute(0, 3, 2, 1)
            uv_tensor1 = c.permute(0, 3, 2, 1)
            uv_tensor = to_normalized(uv_tensor)
            uv_tensor1 = to_normalized(uv_tensor1)
            
            if hybrid == True:
                uv_features = features(uv_tensor.to(device))
                uv_features = uv_features[-1]
                uv_features = proj(uv_features).flatten(2)
                uv_features = linear_1(uv_features)
                uv_features = uv_features.transpose(2, 1)

            elif variant == 'image' and hybrid == False:
                uv_features = features(uv_tensor.to(device))
                uv_features = uv_features.flatten(2).transpose(1, 2)

            elif variant == 'video' and hybrid == False:
                uv_features = features(uv_tensor.to(device))
                uv_features = uv_features.flatten(2)
                uv_features = linear_3(uv_features)
                uv_features = uv_features.transpose(2, 1)

            if variant == 'image' and seq_embed == True:
                uv_features = sequence_embedding(uv_features)

            # 仿照 uv_features 处理逻辑，添加 uv_features2
            if hybrid == True:
                uv_features2 = features(uv_tensor1.to(device))
                uv_features2 = uv_features2[-1]
                uv_features2 = proj(uv_features2).flatten(2)
                (uv_features2.shape)
                uv_features2 = linear_1(uv_features2)
                uv_features2 = uv_features2.transpose(2, 1)

            elif variant == 'image' and hybrid == False:
                uv_features2 = features(uv_tensor1.to(device))
                uv_features2 = uv_features2.flatten(2).transpose(1, 2)

            elif variant == 'video' and hybrid == False:
                uv_features2 = features(uv_tensor1.to(device))
                uv_features2 = uv_features2.flatten(2)
                uv_features2 = linear_3(uv_features2)
                uv_features2 = uv_features2.transpose(2, 1)

            if variant == 'image' and seq_embed == True:
                uv_features2 = sequence_embedding(uv_features2)
            uv_features = torch.cat((uv_features, uv_features2), dim=1)

        
        all_concated_feats = torch.cat((all_concated_feats, uv_features), dim=0)
                            # 释放不再需要的张量
        del uv_tensor, uv_tensor1, uv_features, uv_features2
        torch.cuda.empty_cache()
        return all_concated_feats[1:]