import os
import cv2
import numpy as np
import json
import shutil  # 导入shutil模块用于删除文件夹内容

def load_and_convert_to_tensor(output_folder, label_file):
    """
    将prepare.py脚本的输出转换为(T, H, W, C)格式的张量。
    :param output_folder: prepare.py脚本的输出文件夹路径
    :param label_file: prepare.py脚本生成的标签文件路径
    :return: 一个字典，键为视频名称，值为对应的(T, H, W, C)张量
    """
    # 加载标签文件
    with open(label_file, "r") as f:
        labels = json.load(f)

    video_tensors = {}  # 用于存储每个视频的张量

    for video_name, label in labels.items():
        video_folder = os.path.join(output_folder, label, video_name)
        if not os.path.exists(video_folder):
            # print(f"视频文件夹 {video_folder} 不存在，跳过该视频。")
            continue

        # 获取该视频的所有帧文件
        frame_files = sorted([f for f in os.listdir(video_folder) if f.startswith("mouth_frame_") and f.endswith(".jpg")])

        if not frame_files:
            # print(f"视频文件夹 {video_folder} 中没有找到帧文件，跳过该视频。")
            continue

        # 加载第一帧以获取图像尺寸
        first_frame_path = os.path.join(video_folder, frame_files[0])
        first_frame = cv2.imread(first_frame_path, cv2.IMREAD_GRAYSCALE)
        H, W = first_frame.shape
        C = 1  # 灰度图，通道数为1

        # 初始化张量
        T = len(frame_files)
        video_tensor = np.zeros((T, H, W, C), dtype=np.uint8)

        # 加载所有帧并填充张量
        for idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(video_folder, frame_file)
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            video_tensor[idx, :, :, 0] = frame

        video_tensors[video_name] = video_tensor
        # print(f"视频 {video_name} 转换完成，张量形状为 {video_tensor.shape}")

    return video_tensors


def main():
    output_folder = "dataset/output"  # prepare.py脚本的输出文件夹
    label_file = os.path.join(output_folder, "labels.json")  # prepare.py脚本生成的标签文件

    # 加载标签文件
    with open(label_file, "r") as f:
        labels = json.load(f)

    video_tensors = load_and_convert_to_tensor(output_folder, label_file)

    # 保存张量到对应的子文件夹
    for video_name, tensor in video_tensors.items():
        label = labels[video_name]  # 从标签文件中获取对应的标签
        save_folder = os.path.join(output_folder, label)  # 根据标签确定保存路径
        os.makedirs(save_folder, exist_ok=True)  # 确保保存路径存在
        save_path = os.path.join(save_folder, f"{video_name}.npy")  # 构造保存路径
        np.save(save_path, tensor)  # 保存为Numpy文件
        # print(f"张量已保存到 {save_path}")

        # 删除对应的视频文件夹内容
        video_folder = os.path.join(output_folder, label, video_name)
        shutil.rmtree(video_folder)  # 删除整个文件夹
        # print(f"视频文件夹 {video_folder} 已删除。")


if __name__ == "__main__":
    main()