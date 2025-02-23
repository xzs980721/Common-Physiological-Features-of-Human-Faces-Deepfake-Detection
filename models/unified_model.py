# models/unified_model.py
import torch
import torch.nn as nn
from LipForensics_mix.train_only import *
class UnifiedModel(nn.Module):
    def __init__(self, video_transformer, spatiotemporal_net):
        super(UnifiedModel, self).__init__()
        self.video_transformer = video_transformer
        self.spatiotemporal_net = spatiotemporal_net

    def forward(self, data1,data11,data2,length):
        # 计算 VideoTransformer 的输出
        output = self.video_transformer(data1,data11)

        # 计算 spatiotemporal_net 的输出
        output1 = self.spatiotemporal_net(data2, length)

        return output, output1