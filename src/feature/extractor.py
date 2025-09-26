"""
特征提取模块
使用DINOv2模型提取图像特征
"""
import torch
import torch.nn.functional as F
from typing import Optional


class DinoV2Extractor:
    """DINOv2特征提取器"""
    
    def __init__(self, 
                 model_name: str = "dinov2_vitl14",
                 layer: int = 23,
                 device: str = "cuda"):
        """
        初始化特征提取器
        
        Args:
            model_name: DINOv2模型名称
            layer: 提取特征的层数
            device: 计算设备
        """
        self.model_name = model_name
        self.layer = layer
        self.device = device
        self.model = None
        self._hook_output = None
        
    def init_model(self):
        """延迟初始化模型（节省内存）"""
        if self.model is None:
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
            self.model = self.model.eval().to(self.device)
            # 注册hook以提取中间层特征
            self.hook = self.model.blocks[self.layer].attn.qkv.register_forward_hook(
                lambda m, inp, out: setattr(self, '_hook_output', out))
    
    @torch.no_grad()
    def extract(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        提取图像特征
        
        Args:
            img_tensor: 图像张量 [B, 3, H, W]
        
        Returns:
            特征张量 [B, N_patches, D]
        """
        self.init_model()
        
        # 前向传播
        _ = self.model(img_tensor)
        
        # 从hook获取QKV特征
        qkv = self._hook_output
        B, N, D3 = qkv.shape
        D = D3 // 3
        
        # 提取Value分支
        val = qkv.reshape(B, N, 3, D)[:, :, 2, :]
        # 去除CLS token
        val = val[:, 1:, :]
        
        # L2归一化
        return F.normalize(val, p=2, dim=-1)