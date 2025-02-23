import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from LipForensics_mix.train_only import *
from datetime import datetime
# Import ViT Packages

from models.videotransformer import VideoTransformer
from models.DDFA import *
from utils_ViT import load_pretrained_weights, PRETRAINED_MODELS, as_tuple, resize_positional_embedding_
from models.transformer import *
from dataset_utils.training_dataset_creation import *
from torch import tensor
# Import 3DDFA Packages
import yaml
from TDDFA import TDDFA
import warnings
import re
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from models.unified_model import UnifiedModel 
from LipForensics_mix.models.spatiotemporal_net import get_model

seed = 114514
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 启用 cudnn.benchmark

seed_everything(seed)

paths = []
base_path = r"Video-Transformer-for-Deepfake-Detection-main/FaceForensics++"


data_dir = '/home/ubuntu/Video-Transformer-for-Deepfake-Detection-main/all_data'
# 初始化存储合并数据的列表
all_data1 = []
all_data11 = []
all_data2 = []
all_lengths = []
all_labels = []

# 遍历目录中的所有 .pt 文件
for filename in os.listdir(data_dir):
    
    if filename.endswith('.pt'):
        
        file_path = os.path.join(data_dir, filename)
        # 加载文件内容
        data_dict = torch.load(file_path)
        # 将数据添加到合并列表中
        all_data1.extend(data_dict['data1'])
        all_data11.extend(data_dict['data11'])
        all_data2.extend(data_dict['data2'])
        all_lengths.extend(data_dict['length'])
        all_labels.extend(data_dict['label'])
batch_size =4 # 根据显存调整

# 将数据和标签转换为 numpy 数组
# 将数据和标签转换为 numpy 数组
data = list(zip(all_data1,all_data11, all_data2, all_lengths))
labels = all_labels # 将标签转换为列表

# 计算每个类别的样本数量
ones = sum(1 for label in labels if label == 1.0)
zeros = sum(1 for label in labels if label == 0.0)

print('ones', ones)
print('zeros', zeros)
num_to_sample = zeros
# 确定需要复制的样本数量
if ones < zeros:
    minority_class = 1.0
    majority_class = 0.0

else:
    minority_class = 0.0
    majority_class = 1.0

minority_samples = [(d,l) for d, l in zip(data, labels) if l == minority_class]

# 随机复制少数类别的样本
majority_samples = [(d,l) for d, l in zip(data, labels) if l == majority_class]


resampled_majority_samples = random.choices(minority_samples, k=num_to_sample)+majority_samples

# 合并原始数据和过采样后的数据
data_resampled =[p[0] for p in resampled_majority_samples]
labels_resampled =[p[1] for p in resampled_majority_samples]

# 打印过采样后的样本数量
ones_resampled = sum(1 for label in labels_resampled if label == 1.0)
zeros_resampled = sum(1 for label in labels_resampled if label == 0.0)

print('ones_resampled', ones_resampled)
print('zeros_resampled', zeros_resampled)

# 复原 all_data1, all_data2, all_lengths
all_data1 = [d[0] for d in data_resampled]
all_data11 = [d[1] for d in data_resampled]
all_data2 = [d[2] for d in data_resampled]
all_lengths = [d[3] for d in data_resampled]

# 如果你也有 labels_resampled，可以复原 all_labels
all_labels = labels_resampled


dataset = ProcessedDataset(all_data1,all_data11, all_data2, all_lengths, all_labels)

# 划分数据集为训练集和验证集
train_size = int(0.01 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 创建模型并将其迁移到设备
video_transformer = VideoTransformer('B_16_imagenet1k', pretrained=True, image_size=384, num_classes=1,
                                     seq_embed=True, hybrid=True, variant='video', device=device).to(device)
spatiotemporal_net = get_model().to(device)
model_save_path = 'model.pth'
# 创建统一模型
model = UnifiedModel(video_transformer, spatiotemporal_net).to(device)
if os.path.exists('big1e-4noseeyes.pth'):
    checkpoint = torch.load('big1e-4noseeyes.pth', map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
    print(f"Loaded model, optimizer, and scheduler from {'big1e-4noseeyes.pth'}")
# model.load_state_dict(torch.load(model_save_path))
epochs = 150
lr = 1e-4

# loss function
criterion = nn.BCEWithLogitsLoss().to(device)  # 确保损失函数也在设备上
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# 导入 TensorBoard 相关模块
from torch.utils.tensorboard import SummaryWriter

# 创建带有时间戳的日志目录以确保唯一性
log_dir = os.path.join('runs', f"video_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
writer = SummaryWriter(log_dir)



scaler = torch.amp.GradScaler()

# 定义梯度累积步数
accumulation_steps = 1  # 根据显存和模型大小调整此值
g_uv_coords = DDFA.load_uv_coords(r'Video-Transformer-for-Deepfake-Detection-main/configs/BFM_UV.mat')
validdatalen=(valid_size//batch_size)
traindatalen=(train_size//batch_size)
print(validdatalen,traindatalen)
removelist=[]
# try:
#     for epoch in range(1):
#         acc_now=0
#         epoch_loss = 0
#         epoch_loss1 = 0
#         epoch_accuracy = 0
#         epoch_accuracy_uv = 0
#         epoch_accuracy_lip = 0
#         all_preds = []
#         all_labels = []
#         optimizer.zero_grad()


        

#         # for i, (data1,data11,data2,length, label,idx) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")):
#         #     if(len(label)<batch_size):
#         #         break
#         #     data1 = data1.to(device)
#         #     data11 = data11.to(device)
#         #     data2 = data2.to(device)
#         #     length = length.to(device)
#         #     label = label.float().to(device)
#         #     idx=idx.to(device)
            
#         #     # 前向传播
#         #     output, output1 = model(data1,data11,data2,length, label, device,g_uv_coords,batch_size)
            
#         #     # 计算损失
#         #     loss = criterion(output.squeeze(1), label)
#         #     loss1 = criterion(output1.squeeze(1), label)
#         #     # print('labels',label)
#         #     # print('output:',output)
#         #     # print('loss:',loss.item())
#         #     # print('output1:',output1)
#         #     # print('loss:',loss1.item())

#         #     a = 0.5 
#         #     b = 1 - a
#         #     output2 = a * output + b * output1
#         #     # 反向传播
#         #     scaler.scale(loss).backward()
#         #     scaler.scale(loss1).backward()
#         #     if (i + 1) % accumulation_steps == 0:
#         #         scaler.step(optimizer)
#         #         scaler.update()
#         #         optimizer.zero_grad()


#         #     acc = (abs((output2.squeeze().sigmoid())-label) < 0.5).float().mean()
#         #     acc_now=(acc_now*i+acc)/(i+1)
#         #     # print('acc_now',acc_now)
#         #     epoch_accuracy += acc / traindatalen
#         #     epoch_loss += loss.item() / traindatalen
#         #     epoch_loss1 += loss1.item() / traindatalen
            
#         #     acc_uv = (abs((output.squeeze().sigmoid())-label) < 0.5).float().mean()
#         #     # print('acc_now',acc_now)
#         #     epoch_accuracy_uv += acc_uv / traindatalen
            
#         #     acc_lip = (abs((output1.squeeze().sigmoid())-label) < 0.5).float().mean()
#         #     # print('acc_now',acc_now)
#         #     epoch_accuracy_lip += acc_lip / traindatalen

        
#         # # 记录训练损失和准确率
#         # writer.add_scalar('Train/Loss', epoch_loss, epoch)
#         # writer.add_scalar('Train/Loss1', epoch_loss1, epoch)
#         # writer.add_scalar('Train/Accuracy', epoch_accuracy, epoch)
#         # writer.add_scalar('Train/Accuracy_uv', epoch_accuracy_uv, epoch)
#         # writer.add_scalar('Train/Accuracy_lip', epoch_accuracy_lip, epoch)
#         # torch.save(model.state_dict(), model_save_path)
#         # print(f"Model saved to {model_save_path}")
        
#         with torch.no_grad():
#             epoch_val_accuracy = 0
#             epoch_accuracy_uv = 0
#             epoch_accuracy_lip = 0
#             epoch_val_loss = 0
#             epoch_val_loss1 = 0
#             val_all_preds = []
#             val_all_labels = []
#             for data1,data11,data2,length, label,idx in tqdm(valid_loader, desc=f"Epoch {epoch+1} [Valid]"):
#                 if(len(label)<batch_size):
#                     break
#                 data1 = data1.to(device)
#                 data11 = data11.to(device)
#                 data2 = data2.to(device)
#                 length = length.to(device)
#                 label = label.float().to(device)
#                 # 前向传播
#                 val_output, val_output1 = model(data1,data11,data2,length, label, device,g_uv_coords,batch_size)
#                 # 计算验证损失
#                 val_loss = criterion(val_output.squeeze(1), label).to(device)
#                 val_loss1 = criterion(val_output1.squeeze(1), label).to(device)
                
#                 a = 0.5 
#                 b = 1 - a
#                 val_output2 = a * val_output + b * val_output1
                
#                 acc = (abs(val_output2.squeeze().sigmoid() - label) < 0.5).float().mean()
                
#                 epoch_val_accuracy += acc / validdatalen
#                 epoch_val_loss += val_loss.item() / validdatalen
#                 epoch_val_loss1 += val_loss1.item() / validdatalen
#                 acc_uv = (abs((val_output.squeeze().sigmoid())-label) < 0.5).float().mean()
#                 print('acc',acc)
#                 epoch_accuracy_uv += acc_uv / validdatalen
                
#                 acc_lip = (abs((val_output1.squeeze().sigmoid())-label) < 0.5).float().mean()
#                 # print('acc_now',acc_now)
#                 epoch_accuracy_lip += acc_lip / validdatalen
#                 remove_videos = (abs(val_output.squeeze().sigmoid() - label) >0.6)
#                 for i in range(len(remove_videos)):
#                     if remove_videos[i]:
#                         removelist.append(idx[i])
            


removelist=[tensor(2138), tensor(2084), tensor(2070), tensor(2064), tensor(2050), tensor(2048), tensor(1958), tensor(1940), tensor(1931), tensor(1867), tensor(1853), tensor(1846), tensor(1839), tensor(1838), tensor(1785), tensor(1738), tensor(1731), tensor(1709), tensor(1680), tensor(1679), tensor(1678), tensor(1666), tensor(1649), tensor(1643), tensor(1635), tensor(1625), tensor(1592), tensor(1566), tensor(1550), tensor(1495), tensor(1463), tensor(1443), tensor(1431), tensor(1425), tensor(1422), tensor(1415), tensor(1404), tensor(1395), tensor(1389), tensor(1353), tensor(1350), tensor(1330), tensor(1321), tensor(1268), tensor(1258), tensor(1247), tensor(1239), tensor(1182), tensor(1168), tensor(1151), tensor(1104), tensor(1076), tensor(1075), tensor(1067), tensor(1065), tensor(1060), tensor(1059), tensor(1053), tensor(1049), tensor(1044), tensor(1036), tensor(1032), tensor(1031), tensor(1028), tensor(1025), tensor(1022), tensor(1021), tensor(1007), tensor(1006), tensor(1005), tensor(1004), tensor(994), tensor(991), tensor(988), tensor(986), tensor(985), tensor(982), tensor(980), tensor(979), tensor(977), tensor(975), tensor(974), tensor(972), tensor(970), tensor(968), tensor(959), tensor(958), tensor(956), tensor(951), tensor(950), tensor(944), tensor(941), tensor(938), tensor(936), tensor(935), tensor(926), tensor(922), tensor(917), tensor(911), tensor(903), tensor(893), tensor(892), tensor(891), tensor(890), tensor(889), tensor(886), tensor(885), tensor(884), tensor(873), tensor(872), tensor(870), tensor(867), tensor(863), tensor(862), tensor(852), tensor(847), tensor(846), tensor(841), tensor(833), tensor(831), tensor(825), tensor(821), tensor(819), tensor(816), tensor(815), tensor(812), tensor(811), tensor(809), tensor(807), tensor(784), tensor(782), tensor(775), tensor(770), tensor(768), tensor(765), tensor(752), tensor(751), tensor(748), tensor(742), tensor(738), tensor(734), tensor(726), tensor(724), tensor(712), tensor(708), tensor(703), tensor(701), tensor(700), tensor(697), tensor(694), tensor(692), tensor(685), tensor(679), tensor(664), tensor(661), tensor(654), tensor(648), tensor(647), tensor(644), tensor(641), tensor(640), tensor(635), tensor(634), tensor(633), tensor(626), tensor(624), tensor(616), tensor(615), tensor(605), tensor(604), tensor(603), tensor(601), tensor(600), tensor(599), tensor(592), tensor(589), tensor(587), tensor(580), tensor(575), tensor(570), tensor(568), tensor(567), tensor(565), tensor(558), tensor(554), tensor(543), tensor(539), tensor(537), tensor(536), tensor(534), tensor(530), tensor(529), tensor(525), tensor(518), tensor(511), tensor(503), tensor(501), tensor(497), tensor(490), tensor(489), tensor(479), tensor(467), tensor(454), tensor(448), tensor(440), tensor(436), tensor(435), tensor(431), tensor(429), tensor(422), tensor(420), tensor(412), tensor(409), tensor(406), tensor(401), tensor(396), tensor(394), tensor(389), tensor(388), tensor(379), tensor(375), tensor(373), tensor(366), tensor(361), tensor(357), tensor(352), tensor(348), tensor(334), tensor(327), tensor(323), tensor(321), tensor(317), tensor(316), tensor(315), tensor(309), tensor(305), tensor(304), tensor(303), tensor(302), tensor(301), tensor(295), tensor(293), tensor(286), tensor(278), tensor(276), tensor(273), tensor(266), tensor(256), tensor(252), tensor(251), tensor(245), tensor(240), tensor(238), tensor(230), tensor(224), tensor(223), tensor(222), tensor(218), tensor(210), tensor(207), tensor(199), tensor(198), tensor(197), tensor(189), tensor(188), tensor(183), tensor(182), tensor(173), tensor(172), tensor(170), tensor(165), tensor(162), tensor(161), tensor(159), tensor(154), tensor(143), tensor(126), tensor(124), tensor(110), tensor(109), tensor(103), tensor(102), tensor(96), tensor(93), tensor(90), tensor(88), tensor(86), tensor(83), tensor(80), tensor(78), tensor(63), tensor(61), tensor(60), tensor(58), tensor(55), tensor(44), tensor(32), tensor(31), tensor(27), tensor(25), tensor(22), tensor(20), tensor(16), tensor(8), tensor(6), tensor(4)]
print(f"Removing {removelist}")
torch.save(data_dict, f'data_filtered.pt')
all_data1 = [item for idx, item in enumerate(all_data1) if idx not in removelist]
all_data11 = [item for idx, item in enumerate(all_data11) if idx not in removelist]
all_data2 = [item for idx, item in enumerate(all_data2) if idx not in removelist]
all_lengths = [item for idx, item in enumerate(all_lengths) if idx not in removelist]
all_labels = [item for idx, item in enumerate(all_labels) if idx not in removelist]
data_dict = {
                'data1': all_data1,
                'data11': all_data11,
                'data2': all_data2,
                'length': all_lengths,
                'label': all_labels
            }
torch.save(data_dict, f'data_filtered.pt')

# finally:
#     # 确保在任何情况下都关闭 SummaryWriter
#     writer.close()