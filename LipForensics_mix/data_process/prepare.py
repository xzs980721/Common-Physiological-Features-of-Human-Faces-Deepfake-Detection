import cv2
import dlib
import shutil
import os
import face_recognition
import json
import numpy as np
from tqdm import tqdm  # 引入tqdm库

# 初始化人脸检测器和关键点检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 修改后的关键点索引
MOUTH_POINTS = list(range(48, 68))

def clear_output_dir(output_dir):
    """清空输出目录"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # 删除整个目录及其内容
    os.makedirs(output_dir, exist_ok=True)  # 重新创建目录
    print(f"Output directory '{output_dir}' has been cleared.")


def extract_mouth_region(frame, landmarks):
    """根据关键点裁剪区域"""
    mouth_points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in MOUTH_POINTS])
    x, y, w, h = cv2.boundingRect(mouth_points)
    mouth_region = frame[y:y+h, x:x+w]  # 裁剪彩色图像的区域
    return mouth_region


def process_video(video_path, output_folder, label):
    # 打开video文件
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extract_frame_count = 0
    stride = 3
    frame_list = []

    # 提取前九帧
    while cap.isOpened() and extract_frame_count < 9:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % stride == 0:
            frame_list.append(frame)
            extract_frame_count += 1

    cap.release()

    if len(frame_list) < 9:
        print(f"video {video_path} frame sum <= 9,pass.")
        return

    # 创建输出文件夹
    label_folder = "real" if label == 0 else "fake"  # 根据标签创建对应的文件夹
    video_name = os.path.basename(video_path).split('.')[0]
    video_output_folder = os.path.join(output_folder, label_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    # 遍历提取的帧
    for idx, frame in enumerate(frame_list):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将帧转换为灰度图
        faces = detector(gray)

        if len(faces) == 0:
            print(f"there is no human face in frame {idx} ,pass this frame.")
            continue

        # 假设每帧只处理一个人脸
        face = faces[0]
        landmarks = predictor(gray, face)

        # 裁剪区域（从灰度图中裁剪）
        mouth_region = extract_mouth_region(gray, landmarks)  # 从灰度图中裁剪

        # 统一裁剪区域的大小
        mouth_region_resized = cv2.resize(mouth_region, (96, 96))  # 调整为固定大小

        # 保存裁剪区域
        mouth_output_path = os.path.join(video_output_folder, f"mouth_frame_{idx}.jpg")
        cv2.imwrite(mouth_output_path, mouth_region_resized)

    # 保存标签信息
    label_file = os.path.join(output_folder, "labels.json")
    if not os.path.exists(label_file):
        labels = {}
    else:
        with open(label_file, "r") as f:
            labels = json.load(f)

    labels[video_name] = label_folder  # 记录video名称和对应的标签

    with open(label_file, "w") as f:
        json.dump(labels, f)


def main():
    # 指定输入文件夹和输出文件夹
    input_folder = "dataset/videos"
    output_folder = "dataset/output"

    # 清空输出目录
    clear_output_dir(output_folder)

    # 获取所有视频文件
    video_files = []
    for label, subfolder in enumerate(["real", "fake"]):
        subfolder_path = os.path.join(input_folder, subfolder)
        for filename in os.listdir(subfolder_path):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append((os.path.join(subfolder_path, filename), label))

    # 使用tqdm显示进度
    for video_path, label in tqdm(video_files, desc="Processing videos", unit="video"):
        process_video(video_path, output_folder, label)

    print("Successfully processing videos.")  # 操作结束后显示done


if __name__ == "__main__":
    main()