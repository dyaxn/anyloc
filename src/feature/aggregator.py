"""
特征聚合模块
使用VLAD算法聚合局部特征
"""
import torch
import torch.nn.functional as F
from typing import Optional


class VLADAggregator:
    """VLAD特征聚合器"""
    
    def __init__(self, vocab_path: str, device: str = "cuda"):
        """
        初始化VLAD聚合器
        
        Args:
            vocab_path: 词汇表路径
            device: 计算设备
        """
        self.device = device
        self.centers = self._load_vocabulary(vocab_path)
        self.num_clusters, self.desc_dim = self.centers.shape
    
    def _load_vocabulary(self, vocab_path: str) -> torch.Tensor:
        """加载VLAD词汇表"""
        data = torch.load(vocab_path, map_location='cpu')
        if isinstance(data, dict) and 'c_centers' in data:
            centers = data['c_centers']
        else:
            centers = data
        return centers.to(self.device)
    
    @torch.no_grad()
    def aggregate(self, features: torch.Tensor) -> torch.Tensor:
        """
        VLAD聚合
        
        Args:
            features: 局部特征 [N_patches, D]
        
        Returns:
            VLAD描述符 [K*D]
        """
        features = features.to(self.device)
        
        # 计算到聚类中心的距离
        dists = torch.cdist(features, self.centers, p=2)
        # 硬分配到最近的聚类中心
        assign = torch.argmin(dists, dim=1)
        
        # 初始化VLAD向量
        vlad = torch.zeros(self.num_clusters, self.desc_dim, device=self.device)
        
        # 累积残差
        for k in range(self.num_clusters):
            mask = (assign == k)
            if torch.any(mask):
                residuals = features[mask] - self.centers[k].unsqueeze(0)
                vlad[k] = residuals.sum(dim=0)
        
        # Intra-normalization
        vlad = F.normalize(vlad, p=2, dim=1)
        # 展平
        vlad = vlad.flatten()
        # 最终L2归一化
        vlad = F.normalize(vlad, p=2, dim=0)
        
        return vlad