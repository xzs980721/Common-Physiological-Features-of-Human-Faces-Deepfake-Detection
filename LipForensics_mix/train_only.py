import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from LipForensics_mix.dataset_utils.training_dataset_creation import VideoTrainDataset
from LipForensics_mix.models.spatiotemporal_net import get_model
from torch.nn.utils.rnn import pad_sequence
import argparse
from tqdm import tqdm
import cv2
import dlib  # 使用 dlib 进行人脸特征点检测
from torchvision.transforms.functional import to_tensor, to_pil_image, resize

# 自定义 collate_fn
def collate_fn(batch):
    videos, labels, lengths = zip(*batch)
    videos = pad_sequence(videos, batch_first=True, padding_value=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)  # 使用浮点数标签
    return videos, labels, lengths

# 自定义人脸特征点检测和嘴部区域截取函数
def detect_mouth_region(video_tensor, device):
    # 加载 dlib 的人脸检测器和特征点检测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # 初始化结果张量列表
    mouth_videos = []
    max_get_frames = 9
    # 遍历每一帧，只处理前九帧
    for frame_idx, frame in enumerate(video_tensor.permute(1, 0, 2, 3)):  # 交换维度，变为 (t, c, h, w)
        if frame_idx >= max_get_frames:  # 限制只处理前九帧
            break
        # 将帧从 Tensor 转换为 NumPy 数组
        frame_np = frame.cpu().permute(1, 2, 0).numpy()  # 交换维度，变为 (h, w, c)
        if frame_np.dtype != np.uint8:
            frame_np = (frame_np * 255).astype(np.uint8)  # 将浮点数转换为 0-255 的 uint8
        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)  # 转换为 BGR 格式

        # 检测人脸
        faces = detector(frame_np, 1)
        if len(faces) > 0:
            face = faces[0]  # 取第一个检测到的人脸
            landmarks = predictor(frame_np, face)

            # 提取嘴部区域的特征点
            mouth_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])

            # 计算嘴部区域的边界框
            x, y, w, h = cv2.boundingRect(mouth_points)
            if w <= 0 or h <= 0:
                mouth_region_gray_resized = torch.zeros(1, 48, 48).to(device)  # 假设嘴部区域大小为 48x48 的零张量
            else:
                # 截取嘴部区域
                mouth_region = frame_np[y:y + h, x:x + w]

                # 将嘴部区域转换为灰度图
                mouth_region_gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)

                # 调整图片尺寸为 48x48
                mouth_region_gray_resized = cv2.resize(mouth_region_gray, (48, 48), interpolation=cv2.INTER_LINEAR)

                # 将灰度图转换为 Tensor
                mouth_region_gray_resized = torch.from_numpy(mouth_region_gray_resized).float().to(device)
                mouth_region_gray_resized = mouth_region_gray_resized.unsqueeze(0)  # 添加通道维度
        else:
            # 如果未检测到人脸，返回空张量
            mouth_region_gray_resized = torch.zeros(1, 48, 48).to(device)  # 假设嘴部区域大小为 48x48 的零张量

        mouth_videos.append(mouth_region_gray_resized)

    # 将所有帧合并为一个张量
    mouth_video_tensor = torch.stack(mouth_videos, dim=0)  # 维度为 (t, 1, h, w)
    mouth_video_tensor = mouth_video_tensor.permute(1, 0, 2, 3)  # 交换维度，变为 (1, t, h, w)
    return mouth_video_tensor
def tensor_process(batch, device):
    # 检查 batch 中的值数量
    if len(batch) == 2:
        data, labels = batch
        lengths = torch.tensor([data.size(1)] * data.size(0), dtype=torch.long).to(device)  # 创建一个默认的 lengths 张量
    elif len(batch) == 3:
        data, labels, lengths = batch
        lengths = lengths.to(device)
    else:
        raise ValueError(f"Unexpected number of values in batch: {len(batch)}")

    # 对每个视频张量进行处理
    processed_data = [detect_mouth_region(video, device) for video in data]
    processed_data = pad_sequence(processed_data, batch_first=True, padding_value=0).to(device)

    # 更新 lengths，确保其与处理后的数据一致
    lengths = torch.tensor([min(len(video), 9) for video in processed_data], dtype=torch.long).to(device)  # 假设最多处理 9 帧

    # 将所有数据移动到 GPU
    labels = labels.to(device)

    # 调用模型时始终传递 lengths 参数
    return processed_data, lengths  # 模型输出为 logits