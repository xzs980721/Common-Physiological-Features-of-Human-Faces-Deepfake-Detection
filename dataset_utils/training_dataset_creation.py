import numpy as np
import glob, os
import torch
from .augmentations import *
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import cv2
class ProcessedDataset(Dataset):
    def __init__(self, data1,data11, data2, lengths, labels):
        assert len(data1) == len(data2) == len(lengths) == len(labels), "All input lists must have the same length."
        self.data1 = data1
        self.data11 = data11
        self.data2 = data2
        self.lengths = lengths
        self.labels = labels

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx],self.data11[idx], self.data2[idx], self.lengths[idx], self.labels[idx],idx
class VideoDeepFakeSet(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        video_path = self.file_list[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while cap.isOpened() and frame_count < 9:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
            frame = Image.fromarray(frame)  # 转换为PIL图像
            frame=frame.resize((384,384))
            frame = self.transform(frame)  # 转换为张量
            
            frames.append(frame)
            frame_count += 1
        cap.release()
        
        if len(frames) < 9:
            # 如果帧数少于9帧，跳过该视频
            print(f"Video {video_path} has less than 9 frames. Skipping.")
            return self.__getitem__((idx + 1) % len(self.file_list))  # 尝试获取下一个视频
        
        frames = torch.stack(frames)
        frames = frames.permute(1, 0, 2, 3)  # 调整形状为 (channels, frames, height, width)

        label = video_path.split('/')[-3]
        label = 1 if label == "original_sequences" else 0
        return frames, label

class VideoTrainDataset:
    @staticmethod
    def get_video_batches(paths, batch_size, num_workers=4, pin_memory=False, transform=None, real_limit=3000, fake_limit=750):
        # 获取文件列表
        train_lists = [glob.glob(os.path.join(path, '*')) for path in paths]
        
        # 限制数据集大小
        train_list = []
        train_list.extend(train_lists[0][:real_limit])
        for fake_list in train_lists[1:]:
            train_list.extend(fake_list[:fake_limit])

        # 合并所有数据集
        total_list = train_list

        # 打乱数据集
        np.random.shuffle(total_list)

        # 按4:1的比例分割数据集为训练集和验证集
        split_index = int(0.8 * len(total_list))
        train_list = total_list[:split_index]
        valid_list = total_list[split_index:]

        # 打印数据集信息
        print(f"Total Data Length: {len(total_list)}")
        print(f"Train Data Length: {len(train_list)}")
        print(f"Valid Data Length: {len(valid_list)}")

        # 创建数据集
        train_data = VideoDeepFakeSet(train_list, transform=transform)
        valid_data = VideoDeepFakeSet(valid_list, transform=transform)

        # 创建数据加载器
        train_loader = DataLoader(
            dataset=train_data, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn
        )
        valid_loader = DataLoader(
            dataset=valid_data, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn
        )

        # 打印数据加载器信息
        print(f"Train Loader Length: {len(train_loader)}")
        print(f"Valid Loader Length: {len(valid_loader)}")

        return train_loader, valid_loader

def custom_collate_fn(batch):
    # 过滤掉无效样本
    batch = [data for data in batch if data is not None]
    if not batch:
        return None, None
    
    # 假设 batch 中的每个元素是 (frames, label)
    frames, labels = zip(*batch)
    
    # 将 frames 和 labels 分别堆叠
    frames = torch.stack(frames)
    labels = torch.tensor(labels)
    
    return frames, labels