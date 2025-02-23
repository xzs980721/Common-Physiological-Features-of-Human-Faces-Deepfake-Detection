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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    
    # 计算基础指标
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    # 计算FAR（False Acceptance Rate）
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    far = fp / (fp + tn) if (fp + tn) != 0 else 0
    
    return precision, recall, f1, far
world_size = 2  # 使用2块GPU
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
dist.init_process_group(
    backend='nccl',
    init_method='env://',
    rank=rank,
    world_size=world_size
)
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')
seed = 114514
epochs = 150
lr = 1e-5
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


data_dir = 'all_data_filtered'
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
batch_size =8 # 根据显存调整

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
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
# 创建数据加载器
    # 创建分布式采样器
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=8,
    pin_memory=True,
    drop_last=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    sampler=valid_sampler,
    num_workers=8,
    pin_memory=True,
    drop_last=True
)
# loss function
# 创建模型并将其迁移到设备
video_transformer = VideoTransformer('B_16_imagenet1k', pretrained=True, image_size=384, num_classes=1,
                                     seq_embed=True, hybrid=True, variant='video', device=device).to(device)
spatiotemporal_net = get_model().to(device)
model_init_path = '/home/ubuntu/Video-Transformer-for-Deepfake-Detection-main/1e-5.pth'
model_save_path = 'model.pth'
# 创建统一模型
model = UnifiedModel(video_transformer, spatiotemporal_net).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)  # 确保损失函数也在设备上
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# if os.path.exists(model_init_path):
#     checkpoint = torch.load(model_init_path, map_location=device)
#     model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
#     # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#     print(f"Loaded model, optimizer, and scheduler from {model_init_path}")
model = DDP(model, device_ids=[local_rank],find_unused_parameters=True)
# model.load_state_dict(torch.load(model_save_path))



# 导入 TensorBoard 相关模块
from torch.utils.tensorboard import SummaryWriter

# 创建带有时间戳的日志目录以确保唯一性
log_dir = os.path.join('/home/ubuntu/Video-Transformer-for-Deepfake-Detection-main/xjtudeepfakedata/runs', f"video_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
# 只在主进程初始化TensorBoard
writer = SummaryWriter(log_dir)



scaler = torch.amp.GradScaler()

# 定义梯度累积步数
accumulation_steps = 1  # 根据显存和模型大小调整此值
g_uv_coords = DDFA.load_uv_coords(r'Video-Transformer-for-Deepfake-Detection-main/configs/BFM_UV.mat')
validdatalen=(len(valid_loader))
traindatalen=(len(train_loader))
print('validdatalen',traindatalen)
print('traindatalen',validdatalen)
removelist=[]
try:
    for epoch in range(epochs):

        train_sampler.set_epoch(epoch)
        acc_now=0
        epoch_loss = 0
        epoch_loss1 = 0
        epoch_accuracy = 0
        epoch_accuracy_uv = 0
        epoch_accuracy_lip = 0
        all_preds = []
        all_labels = []
        optimizer.zero_grad()


        
        
        for i, (data1,data11,data2,length, label,idx) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")):
            if(len(label)<batch_size):
                traindatalen-=1
                break
            data1 = data1.to(device)
            data11 = data11.to(device)
            data2 = data2.to(device)
            length = length.to(device)
            label = label.float().to(device)
            idx=idx.to(device)
            train_metrics=0,0,0,0
            # 前向传播
            with autocast():
                output, output1 = model(data1,data11,data2,length)
                # 计算损失
                loss = criterion(output.squeeze(1), label)
                loss1 = criterion(output1.squeeze(1), label)
            # print('labels',label)
            # print('output:',output)
            # print('loss:',loss.item())
            # print('output1:',output1)
            # print('loss:',loss1.item())

            a = 0.5 
            b = 1 - a
            output2 = a * output + b * output1
            # 反向传播
            loss2=loss1+loss
            scaler.scale(loss2).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()


            acc = (abs((output2.squeeze().sigmoid())-label) < 0.5).float().mean()
            acc_now=(acc_now*i+acc)/(i+1)
            # print('acc_now',acc_now)
            epoch_accuracy += acc / traindatalen
            epoch_loss += loss.item() / traindatalen
            epoch_loss1 += loss1.item() / traindatalen
            
            acc_uv = (abs((output.squeeze().sigmoid())-label) < 0.5).float().mean()
            # print('acc_now',acc_now)
            epoch_accuracy_uv += acc_uv / traindatalen
            
            acc_lip = (abs((output1.squeeze().sigmoid())-label) < 0.5).float().mean()
            # print('acc_now',acc_now)
            epoch_accuracy_lip += acc_lip / traindatalen
            preds = (output2.squeeze().sigmoid() > 0.5).float().cpu().numpy()
            labels_cpu = label.cpu().numpy()

            # 收集结果
            all_preds.extend(preds.tolist())
            all_labels.extend(labels_cpu.tolist())
        
        train_metrics=calculate_metrics(all_preds, all_labels)
        

        
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_accuracy1=0
            epoch_val_accuracy2=0
            epoch_val_accuracy_uv = 0
            epoch_val_accuracy_lip = 0
            epoch_val_loss = 0
            epoch_val_loss1 = 0
            val_all_preds = []
            val_all_preds_uv=[]
            val_all_preds_lip=[]
            val_all_labels = []
            for i,(data1,data11,data2,length, label,idx) in enumerate(tqdm(valid_loader, desc=f"Epoch {epoch+1} [Valid]")):
                
                if(len(label)<batch_size):
                    validdatalen-=1
                    break
                data1 = data1.to(device)
                data11 = data11.to(device)
                data2 = data2.to(device)
                length = length.to(device)
                label = label.float().to(device)
                # 前向传播
                val_output, val_output1 = model(data1,data11,data2,length)
                # 计算验证损失
                val_loss = criterion(val_output.squeeze(1), label).to(device)
                val_loss1 = criterion(val_output1.squeeze(1), label).to(device)
                
                a = 0.5 
                b = 1 - a
                val_output2 = a * val_output + b * val_output1
                val_output3 = 0.3 * val_output + 0.7 * val_output1
                val_output4 = 0.7 * val_output + 0.3 * val_output1
                ##指标计算
                acc = (abs(val_output2.squeeze().sigmoid() - label) < 0.5).float().mean()
                acc1 = (abs(val_output3.squeeze().sigmoid() - label) < 0.5).float().mean()
                acc2 = (abs(val_output4.squeeze().sigmoid() - label) < 0.5).float().mean()
                epoch_val_accuracy += acc / validdatalen
                epoch_val_accuracy1 += acc1 / validdatalen
                epoch_val_accuracy2 += acc2 / validdatalen
                epoch_val_loss += val_loss.item() / validdatalen
                epoch_val_loss1 += val_loss1.item() / validdatalen
                acc_uv = (abs((val_output.squeeze().sigmoid())-label) < 0.5).float().mean()
                # print('acc_now',acc_now)
                epoch_val_accuracy_uv += acc_uv / validdatalen
                
                acc_lip = (abs((val_output1.squeeze().sigmoid())-label) < 0.5).float().mean()
                # print('acc_now',acc_now)
                epoch_val_accuracy_lip += acc_lip / validdatalen
                preds = (val_output2.squeeze().sigmoid() > 0.5).float().cpu().numpy()
                preds_uv = (val_output.squeeze().sigmoid() > 0.5).float().cpu().numpy()
                preds_lip = (val_output1.squeeze().sigmoid() > 0.5).float().cpu().numpy()
                labels_cpu = label.cpu().numpy()

                # 收集结果
                val_all_preds.extend(preds.tolist())
                val_all_preds_uv.extend(preds_uv.tolist())
                val_all_preds_lip.extend(preds_lip.tolist())
                val_all_labels.extend(labels_cpu.tolist())
                
            val_metrics=calculate_metrics(val_all_preds, val_all_labels)
            val_metrics_uv=calculate_metrics(val_all_preds_uv, val_all_labels)
            val_metrics_lip=calculate_metrics(val_all_preds_lip, val_all_labels)
                
            
            # 记录验证损失和准确率
            # 同步验证损失和准确率
        val_precision_tensor = torch.as_tensor(val_metrics[0], device=device)
        val_recall_tensor = torch.as_tensor(val_metrics[1], device=device)
        val_f1_tensor = torch.as_tensor(val_metrics[2], device=device)
        val_far_tensor = torch.as_tensor(val_metrics[3], device=device)
        val_precision_tensor_uv = torch.as_tensor(val_metrics_uv[0], device=device)
        val_recall_tensor_uv = torch.as_tensor(val_metrics_uv[1], device=device)
        val_f1_tensor_uv = torch.as_tensor(val_metrics_uv[2], device=device)
        val_far_tensor_uv = torch.as_tensor(val_metrics_uv[3], device=device)
        val_precision_tensor_lip = torch.as_tensor(val_metrics_lip[0], device=device)
        val_recall_tensor_lip = torch.as_tensor(val_metrics_lip[1], device=device)
        val_f1_tensor_lip = torch.as_tensor(val_metrics_lip[2], device=device)
        val_far_tensor_lip = torch.as_tensor(val_metrics_lip[3], device=device)
        # 同步所有进程的指标
        torch.distributed.all_reduce(val_precision_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_recall_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_f1_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_far_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_precision_tensor_uv, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_recall_tensor_uv, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_f1_tensor_uv, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_far_tensor_uv, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_precision_tensor_lip,op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_recall_tensor_lip, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_f1_tensor_lip, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(val_far_tensor_lip, op=torch.distributed.ReduceOp.SUM)
        val_metrics = (
            val_precision_tensor.item() / world_size,
            val_recall_tensor.item() / world_size,
            val_f1_tensor.item() / world_size,
            val_far_tensor.item() / world_size
        )
        val_metrics_uv = (
            val_precision_tensor_uv.item() / world_size,
            val_recall_tensor_uv.item() / world_size,
            val_f1_tensor_uv.item() / world_size,
            val_far_tensor_uv.item() / world_size
        )
        val_metrics_lip = (
            val_precision_tensor_lip.item() / world_size,
            val_recall_tensor_lip.item() / world_size,
            val_f1_tensor_lip.item() / world_size,
            val_far_tensor_lip.item() / world_size
        )
        precision_tensor = torch.as_tensor(train_metrics[0], device=device)
        recall_tensor = torch.as_tensor(train_metrics[1], device=device)
        f1_tensor = torch.as_tensor(train_metrics[2], device=device)
        far_tensor = torch.as_tensor(train_metrics[3], device=device)

        # 同步所有进程的指标
        torch.distributed.all_reduce(precision_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(recall_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(f1_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(far_tensor, op=torch.distributed.ReduceOp.SUM)
        train_metrics = (
            precision_tensor.item() / world_size,
            recall_tensor.item() / world_size,
            f1_tensor.item() / world_size,
            far_tensor.item() / world_size
        )
        epoch_val_loss_tensor = torch.as_tensor(epoch_val_loss, device=device)
        epoch_val_loss1_tensor = torch.as_tensor(epoch_val_loss1, device=device)
        epoch_val_accuracy_tensor = torch.as_tensor(epoch_val_accuracy, device=device)
        epoch_val_accuracy_tensor1 = torch.as_tensor(epoch_val_accuracy1, device=device)
        epoch_val_accuracy_tensor2 = torch.as_tensor(epoch_val_accuracy2, device=device)
        epoch_val_accuracy_uv_tensor = torch.as_tensor(epoch_val_accuracy_uv, device=device)
        epoch_val_accuracy_lip_tensor = torch.as_tensor(epoch_val_accuracy_lip, device=device)
        dist.all_reduce(epoch_val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_val_loss1_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_val_accuracy_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_val_accuracy_tensor1, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_val_accuracy_tensor2, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_val_accuracy_uv_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_val_accuracy_lip_tensor, op=dist.ReduceOp.SUM)
        # 计算平均值
        epoch_val_loss = epoch_val_loss_tensor.item() / dist.get_world_size()
        epoch_val_loss1 = epoch_val_loss1_tensor.item() / dist.get_world_size()
        epoch_val_accuracy = epoch_val_accuracy_tensor.item() / dist.get_world_size()
        epoch_val_accuracy1 = epoch_val_accuracy_tensor1.item() / dist.get_world_size()
        epoch_val_accuracy2 = epoch_val_accuracy_tensor2.item() / dist.get_world_size()
        epoch_val_accuracy_uv = epoch_val_accuracy_uv_tensor.item() / dist.get_world_size()
        epoch_val_accuracy_lip = epoch_val_accuracy_lip_tensor.item() / dist.get_world_size()
        
        
        epoch_loss_tensor = torch.as_tensor(epoch_loss, device=device)
        epoch_loss1_tensor = torch.as_tensor(epoch_loss1, device=device)
        epoch_accuracy_tensor = torch.as_tensor(epoch_accuracy, device=device)
        epoch_accuracy_uv_tensor = torch.as_tensor(epoch_accuracy_uv, device=device)
        epoch_accuracy_lip_tensor = torch.as_tensor(epoch_accuracy_lip, device=device)

        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_loss1_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_accuracy_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_accuracy_uv_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_accuracy_lip_tensor, op=dist.ReduceOp.SUM)

        # 计算平均值
        epoch_loss = epoch_loss_tensor.item() / dist.get_world_size()
        epoch_loss1 = epoch_loss1_tensor.item() / dist.get_world_size()
        epoch_accuracy = epoch_accuracy_tensor.item() / dist.get_world_size()
        epoch_accuracy_uv = epoch_accuracy_uv_tensor.item() / dist.get_world_size()
        epoch_accuracy_lip = epoch_accuracy_lip_tensor.item() / dist.get_world_size()
        # 记录验证损失和准确率
        if rank == 0:
            writer.add_scalar('Validation/Loss', epoch_val_loss, epoch)
            writer.add_scalar('Validation/Loss1', epoch_val_loss1, epoch)
            writer.add_scalar('Validation/Accuracy', epoch_val_accuracy, epoch)
            writer.add_scalar('Validation/Accuracy1', epoch_val_accuracy1, epoch)
            writer.add_scalar('Validation/Accuracy2', epoch_val_accuracy2, epoch)
            writer.add_scalar('Validation/Accuracy_uv', epoch_val_accuracy_uv, epoch)
            writer.add_scalar('Validation/Accuracy_lip', epoch_val_accuracy_lip, epoch)
            writer.add_scalar('Train/Loss', epoch_loss, epoch)
            writer.add_scalar('Train/Loss1', epoch_loss1, epoch)
            writer.add_scalar('Train/Accuracy', epoch_accuracy, epoch)
            writer.add_scalar('Train/Accuracy_uv', epoch_accuracy_uv, epoch)
            writer.add_scalar('Train/Accuracy_lip', epoch_accuracy_lip, epoch)
            writer.add_scalars('Metrics/Val', {
                'precision': val_metrics[0],
                'recall': val_metrics[1],
                'f1': val_metrics[2],
                'far': val_metrics[3]
            }, epoch)
            writer.add_scalars('Metrics/Val_uv', {
                'precision': val_metrics_uv[0],
                'recall': val_metrics_uv[1],
                'f1': val_metrics_uv[2],
                'far': val_metrics_uv[3]
            }, epoch)
            writer.add_scalars('Metrics/Val_lip', {
                'precision': val_metrics_lip[0],
                'recall': val_metrics_lip[1],
                'f1': val_metrics_lip[2],
                'far': val_metrics_lip[3]
            }, epoch)
            # writer.add_scalars('Metrics/Train', {
            #     'precision': train_metrics[0],
            #     'recall': train_metrics[1],
            #     'f1': train_metrics[2],
            #     'far': train_metrics[3]
            # }, epoch)
        torch.save({
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),
        }, model_save_path)
        print(f"Model saved to {model_save_path}")
        # scheduler.step()
        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
finally:
    dist.destroy_process_group()
    # 确保在任何情况下都关闭 SummaryWriter
    writer.close()
