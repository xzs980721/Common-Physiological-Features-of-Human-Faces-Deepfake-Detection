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
from torch.utils.tensorboard import SummaryWriter
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
from torch import tensor
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

device = torch.device(f'cuda:0')
seed = 114514
epochs = 150
lr = 1e-6
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

# 创建模型并将其迁移到设备
video_transformer = VideoTransformer('B_16_imagenet1k', pretrained=True, image_size=384, num_classes=1,
                                     seq_embed=True, hybrid=True, variant='video', device=device).to(device)
spatiotemporal_net = get_model().to(device)
model_init_path = '/home/ubuntu/Video-Transformer-for-Deepfake-Detection-main/1e-5.pth'
model_save_path = 'model.pth'

# 创建统一模型
model = UnifiedModel(video_transformer, spatiotemporal_net).to(device)
# optimizer
file_path = '/home/ubuntu/Video-Transformer-for-Deepfake-Detection-main/all_data/all_data_epoch0_batch100.pt'
# 加载文件内容
data_dict = torch.load(file_path)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# if os.path.exists(model_init_path):
#     checkpoint = torch.load(model_init_path, map_location=device)
#     model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
#     # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#     print(f"Loaded model, optimizer, and scheduler from {model_init_path}")
data1 = torch.rand([1,9,299,299,3]).to(device)
data11 = torch.rand([1,9,299,299,3]).to(device)
data2 = torch.rand([1,1,9,48,48]).to(device)
length=torch.rand([1,1]).int().to(device)

device=device
batch_size=16

g_uv_coords = DDFA.load_uv_coords(r'Video-Transformer-for-Deepfake-Detection-main/configs/BFM_UV.mat')

with SummaryWriter("./log", comment="sample_model_visualization") as sw:
    sw.add_graph(model, (data1,data11,data2,length))
# model.load_state_dict(torch.load(model_save_path))

