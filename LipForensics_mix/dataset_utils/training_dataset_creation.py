import numpy as np
import glob, os
import torch
from .augmentations import *
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import cv2


class ImageDeepFakeSet(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split('\\')[-2].split('.')[0].split('/')[-1]
        label = 1 if label == "real" else 0

        return img_transformed, label


class ImageTrainDataset:
    def get_image_batches(paths, batch_size):
        batch_size = batch_size

        train_list_real = glob.glob(os.path.join(paths[0], '*'))
        train_list_fake = glob.glob(os.path.join(paths[1], '*'))
        train_list_fake_2 = glob.glob(os.path.join(paths[2], '*'))
        train_list_fake_3 = glob.glob(os.path.join(paths[3], '*'))
        train_list_fake_4 = glob.glob(os.path.join(paths[4], '*'))

        valid_list_real = glob.glob(os.path.join(paths[5], '*'))
        valid_list_fake = glob.glob(os.path.join(paths[6], '*'))
        valid_list_fake_2 = glob.glob(os.path.join(paths[7], '*'))
        valid_list_fake_3 = glob.glob(os.path.join(paths[8], '*'))
        valid_list_fake_4 = glob.glob(os.path.join(paths[9], '*'))

        train_list = []
        train_list.extend(train_list_real[:100000])
        train_list.extend(train_list_fake[:25000])
        train_list.extend(train_list_fake_2[:25000])
        train_list.extend(train_list_fake_3[:25000])
        train_list.extend(train_list_fake_4[:25000])

        valid_list = []
        valid_list.extend(valid_list_real[:20000])
        valid_list.extend(valid_list_fake[:5000])
        valid_list.extend(valid_list_fake_2[:5000])
        valid_list.extend(valid_list_fake_3[:5000])
        valid_list.extend(valid_list_fake_4[:5000])

        print(f"Train Data Real: {len(train_list_real)}")
        print(f"Train Data Fake: {len(train_list_fake)}")
        print(f"Train Data Fake 2: {len(train_list_fake_2)}")
        print(f"Train Data Fake 3: {len(train_list_fake_3)}")
        print(f"Train Data Fake 4: {len(train_list_fake_4)}")

        np.random.shuffle(train_list)
        np.random.shuffle(train_list)
        np.random.shuffle(train_list)
        np.random.shuffle(train_list)

        np.random.shuffle(valid_list)
        np.random.shuffle(valid_list)
        np.random.shuffle(valid_list)
        np.random.shuffle(valid_list)

        labels = [path.split('\\')[-2].split('.')[0].split('/')[-1] for path in train_list]
        print("Labels: " + str(labels[:10]))

        train_data = ImageDeepFakeSet(train_list, transform=transforms_imgaug)
        valid_data = ImageDeepFakeSet(valid_list, transform=val_transforms)

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)

        print(len(train_data), len(train_loader))
        print(len(valid_data), len(valid_loader))

        return train_loader, valid_loader


class VideoDeepFakeSet(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        frames_in_video = sorted(glob.glob(self.file_list[idx] + '/*.png'))
        concated_images_per_video = torch.zeros(1, 3, 299, 299)
        for i in range(9):
            img_path = frames_in_video[i]
            img = Image.open(img_path)
            img_transformed = self.transform(img)
            img_transformed = img_transformed.unsqueeze(0)
            concated_images_per_video = torch.cat((concated_images_per_video, img_transformed), dim=0)

        label = img_path.split('\\')[-2].split('.')[0].split('/')[-1]
        label = 1 if label == "real" else 0

        return concated_images_per_video[1:], label


class VideoTrainDataset:
    def get_image_batches(paths, batch_size):
        batch_size = batch_size

        train_list_real = glob.glob(os.path.join(paths[0], '*'))
        train_list_fake = glob.glob(os.path.join(paths[1], '*'))
        train_list_fake_2 = glob.glob(os.path.join(paths[2], '*'))
        train_list_fake_3 = glob.glob(os.path.join(paths[3], '*'))
        train_list_fake_4 = glob.glob(os.path.join(paths[4], '*'))

        valid_list_real = glob.glob(os.path.join(paths[5], '*'))
        valid_list_fake = glob.glob(os.path.join(paths[6], '*'))
        valid_list_fake_2 = glob.glob(os.path.join(paths[7], '*'))
        valid_list_fake_3 = glob.glob(os.path.join(paths[8], '*'))
        valid_list_fake_4 = glob.glob(os.path.join(paths[9], '*'))

        train_list = []
        train_list.extend(train_list_real[:3000])
        train_list.extend(train_list_fake[:750])
        train_list.extend(train_list_fake_2[:750])
        train_list.extend(train_list_fake_3[:750])
        train_list.extend(train_list_fake_4[:750])

        valid_list = []
        valid_list.extend(valid_list_real[:600])
        valid_list.extend(valid_list_fake[:150])
        valid_list.extend(valid_list_fake_2[:150])
        valid_list.extend(valid_list_fake_3[:150])
        valid_list.extend(valid_list_fake_4[:150])

        print(f"Train Data Real: {len(train_list_real)}")
        print(f"Train Data Fake: {len(train_list_fake)}")
        print(f"Train Data Fake 2: {len(train_list_fake_2)}")
        print(f"Train Data Fake 3: {len(train_list_fake_3)}")
        print(f"Train Data Fake 4: {len(train_list_fake_4)}")

        np.random.shuffle(train_list)
        np.random.shuffle(train_list)
        np.random.shuffle(train_list)
        np.random.shuffle(train_list)

        np.random.shuffle(valid_list)
        np.random.shuffle(valid_list)
        np.random.shuffle(valid_list)
        np.random.shuffle(valid_list)

        labels = [path.split('\\')[-2].split('.')[0].split('/')[-1] for path in train_list]
        print("Labels: " + str(labels[:10]))

        train_data = ImageDeepFakeSet(train_list, transform=transforms_imgaug)
        valid_data = ImageDeepFakeSet(valid_list, transform=val_transforms)

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)

        print(len(train_data), len(train_loader))
        print(len(valid_data), len(valid_loader))

        return train_loader, valid_loader


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
            frame = self.transform(frame)  # 转换为张量
            frames.append(frame)
            frame_count += 1
        cap.release()
        if len(frames) == 0:
            raise RuntimeError(f"No frames found in video: {video_path}")
        frames = torch.stack(frames)
        frames = frames.permute(1, 0, 2, 3)  # 调整形状为 (channels, frames, height, width)
        label = video_path.split('\\')[-2].split('.')[0].split('/')[-1].split('_')[1]
        # label = video_path.split('\\')[-2].split('.')[0].split('/')[-1]  #原来的文件
        label = 1 if label == "real" else 0
        return frames, label


class VideoTrainDataset:
    @staticmethod
    def get_video_batches(paths, batch_size, transform=None):
        train_list_real = glob.glob(os.path.join(paths[0], '*'))
        train_list_fake = glob.glob(os.path.join(paths[1], '*'))
        train_list_fake_2 = glob.glob(os.path.join(paths[2], '*'))
        train_list_fake_3 = glob.glob(os.path.join(paths[3], '*'))
        train_list_fake_4 = glob.glob(os.path.join(paths[4], '*'))

        valid_list_real = glob.glob(os.path.join(paths[5], '*'))
        valid_list_fake = glob.glob(os.path.join(paths[6], '*'))
        valid_list_fake_2 = glob.glob(os.path.join(paths[7], '*'))
        valid_list_fake_3 = glob.glob(os.path.join(paths[8], '*'))
        valid_list_fake_4 = glob.glob(os.path.join(paths[9], '*'))

        train_list = []
        train_list.extend(train_list_real[:3000])
        train_list.extend(train_list_fake[:750])
        train_list.extend(train_list_fake_2[:750])
        train_list.extend(train_list_fake_3[:750])
        train_list.extend(train_list_fake_4[:750])

        valid_list = []
        valid_list.extend(valid_list_real[:600])
        valid_list.extend(valid_list_fake[:150])
        valid_list.extend(valid_list_fake_2[:150])
        valid_list.extend(valid_list_fake_3[:150])
        valid_list.extend(valid_list_fake_4[:150])

        print(f"Train Data Real: {len(train_list_real)}")
        print(f"Train Data Fake: {len(train_list_fake)}")
        print(f"Train Data Fake 2: {len(train_list_fake_2)}")
        print(f"Train Data Fake 3: {len(train_list_fake_3)}")
        print(f"Train Data Fake 4: {len(train_list_fake_4)}")

        np.random.shuffle(train_list)
        np.random.shuffle(train_list)
        np.random.shuffle(train_list)
        np.random.shuffle(train_list)

        np.random.shuffle(valid_list)
        np.random.shuffle(valid_list)
        np.random.shuffle(valid_list)
        np.random.shuffle(valid_list)

        train_data = VideoDeepFakeSet(train_list, transform=transform)
        valid_data = VideoDeepFakeSet(valid_list, transform=transform)

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=0)

        print(len(train_data), len(train_loader))
        print(len(valid_data), len(valid_loader))

        return train_loader, valid_loader