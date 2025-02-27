"""model.py - Model and module class for ViT.
   They are built to mirror those in the official Jax implementation.
"""

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import timm
from torch.utils import model_zoo
from functools import reduce
import pretrainedmodels

from vit_configs.configs import PRETRAINED_MODELS, load_pretrained_weights, as_tuple, resize_positional_embedding_
from transformer import *
# Load model directly
from transformers import AutoModel

# Import 3DDFA Packages
from models.DDFA import *


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x
    

class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.pos_embedding
    
class SequenceEmbedding(nn.Module):
    """Adds (optionally learned) sequence embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.seq_embedding = nn.Parameter(torch.zeros(1, 64, dim))
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        return x + self.seq_embedding    


class VideoTransformer(nn.Module):
    """
    Args:
        name (str): Model name, e.g. 'B_16'
        pretrained (bool): Load pretrained weights
        in_channels (int): Number of channels in input data
        num_classes (int): Number of classes, default 1000

    References:
        [1] https://openreview.net/forum?id=YicbFdNTTy
    """

    def __init__(
        self, 
        name: Optional[str] = None, 
        pretrained: bool = False, 
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1,
        representation_size: Optional[int] = None,
        load_repr_layer: bool = False,
        classifier: str = 'token',
        positional_embedding: str = '1d',
        in_channels: int = 3, 
        image_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        seq_embed: bool = True, 
        hybrid: bool = True,
        variant: str = 'video',
        device: str = 'cpu'
    ):
        super().__init__()

        # Configuration
        if name is None:
            check_msg = 'must specify name of pretrained model'
            assert not pretrained, check_msg
            if num_classes is None:
                num_classes = 1000
            if image_size is None:
                image_size = 384
        else:  # load pretrained model
            assert name in PRETRAINED_MODELS.keys(), \
                'name should be in: ' + ', '.join(PRETRAINED_MODELS.keys())
            config = PRETRAINED_MODELS[name]['config']
            patches = config['patches']
            dim = config['dim']
            ff_dim = config['ff_dim']
            num_heads = config['num_heads']
            num_layers = config['num_layers']
            attention_dropout_rate = config['attention_dropout_rate']
            dropout_rate = config['dropout_rate']
            representation_size = config['representation_size']
            classifier = config['classifier']
            if image_size is None:
                image_size = PRETRAINED_MODELS[name]['image_size']
            if num_classes is None:
                num_classes = PRETRAINED_MODELS[name]['num_classes']
        self.seq_embed = seq_embed
        self.hybrid = hybrid
        self.image_size = image_size
        self.device = device
        self.variant = variant
        # Image and patch sizes
        h, w = as_tuple(image_size)  # image sizes
        fh, fw = as_tuple(patches)  # patch sizes
        gh, gw = h // fh, w // fw  # number of patches
        seq_len = gh * gw
        
        if self.hybrid == False:
            # Patch embedding
            self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(16, 20), stride=(16, 20))
        else:
           
            # 创建 Xception41p 模型，不使用预训练权重
            self.xception_feature_extractor = timm.create_model('xception41p.ra3_in1k', features_only=True, pretrained=False, in_chans=in_channels)

            # 手动加载本地预训练权重文件
            pretrained_state_dict = torch.load('Video-Transformer-for-Deepfake-Detection-main/models/pytorch_model.bin')
            # 修改前缀处理方式，可能需要保留部分前缀
            new_state_dict = {}
            for key, value in pretrained_state_dict.items():
                new_key = key.replace('blocks.', 'blocks_')  # 将blocks.0转换为blocks_0
                new_key = new_key.replace('stem.', 'stem_')
                new_state_dict[new_key] = value
            new_state_dict.pop('head.fc.weight', None)
            new_state_dict.pop('head.fc.bias', None)
            self.xception_feature_extractor.load_state_dict(new_state_dict)
        self.sequence_embedding = SequenceEmbedding(64, dim)
        self.proj = nn.Conv2d(2048, 768, 1)
        self.linear_1 = torch.nn.Linear(100, 32)
        self.linear_3 = torch.nn.Linear(252, 32)
                
        # Class token
        if classifier == 'token':
            self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
            seq_len += 1
        
        # Positional embedding
        if positional_embedding.lower() == '1d':
            self.positional_embedding = PositionalEmbedding1D(seq_len, dim)
        else:
            raise NotImplementedError()
                    
        # Transformer
        self.transformer = Transformer(num_layers=num_layers, dim=dim, num_heads=num_heads, 
                                       ff_dim=ff_dim, dropout=dropout_rate)
        
        # Representation layer
        if representation_size and load_repr_layer:
            self.pre_logits = nn.Linear(dim, representation_size)
            pre_logits_size = representation_size
        else:
            pre_logits_size = dim

        # Classifier head
        self.norm = nn.LayerNorm(pre_logits_size, eps=1e-6)
        self.fc = nn.Linear(pre_logits_size, num_classes)

        # Initialize weights
        self.init_weights()
        

        
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.normal_(self.positional_embedding.pos_embedding, std=0.02)  # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.constant_(self.class_token, 0)

    def forward(self, a, c):
        # a = x.transpose(1, 2)
        batch, num_images, channels, height, width = a.shape
        if self.hybrid == False:
            # Patch embedding based VideoTransformer
            video_frames = []
            for batch_number in range(batch):
                all_frames = []
                for images in range((num_images)):
                    x_out = DDFA.generate_uv_tex(a[batch_number][images].unsqueeze(0), c[batch_number][images].unsqueeze(0), self.patch_embedding, self.sequence_embedding, self.linear_1, self.linear_3, self.proj, self.seq_embed, self.hybrid, self.variant, self.device)
                    all_frames.append(x_out)
                images_concated = reduce(lambda n, m: torch.cat((n, m), dim=1), all_frames[:])
                video_frames.append(images_concated)
            batch_images_cat = reduce(lambda n, m: torch.cat((n, m), dim=0), video_frames[:])

        else:
            # Hybrid VideoTransformer
            video_frames = []
            for batch_number in range(batch):
                all_frames = []
                for images in range((num_images)):
                    x_out = DDFA.generate_uv_tex(a[batch_number][images].unsqueeze(0), c[batch_number][images].unsqueeze(0), self.xception_feature_extractor, self.sequence_embedding, self.linear_1, self.linear_3, self.proj, self.seq_embed, self.hybrid, self.variant, self.device)
                    # Ensure x_out is 4D
                    x_out = self.sequence_embedding(x_out)
                    all_frames.append(x_out)
                images_concated = reduce(lambda n, m: torch.cat((n, m), dim=1), all_frames[:])
                video_frames.append(images_concated)
            batch_images_cat = reduce(lambda n, m: torch.cat((n, m), dim=0), video_frames[:])
        
        if hasattr(self, 'class_token'):
            x = torch.cat((self.class_token.expand(batch, -1, -1), batch_images_cat), dim=1)  # b,gh*gw+1,d
        if hasattr(self, 'positional_embedding'): 
            x = self.positional_embedding(x)  # b,gh*gw+1,d 

        x = self.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0]  # b,d
            x = self.fc(x)  # b,num_classes
        return x


