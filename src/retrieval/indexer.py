"""
索引管理模块
管理FAISS索引的构建和查询
"""
import faiss
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class FAISSIndexer:
    """FAISS索引管理器"""
    
    def __init__(self, dimension: int, use_gpu: bool = True):
        """
        初始化索引器
        
        Args:
            dimension: 特征维度
            use_gpu: 是否使用GPU加速
        """
        self.dimension = dimension
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.index = None
        
    def build_index(self, descriptors: torch.Tensor):
        """
        构建索引
        
        Args:
            descriptors: 数据库描述符 [N, D]
        """
        # L2归一化（用于余弦相似度）
        descriptors_norm = F.normalize(descriptors, p=2, dim=1)
        descriptors_np = descriptors_norm.numpy().astype('float32')
        
        # 创建内积索引（归一化后等价于余弦相似度）
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # GPU加速
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # 添加数据
        self.index.add(descriptors_np)
    
    def search(self, query: torch.Tensor, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        检索最相似的向量
        
        Args:
            query: 查询描述符 [1, D]
            top_k: 返回top-k个结果
        
        Returns:
            (distances, indices): 相似度分数和索引
        """
        # L2归一化
        query_norm = F.normalize(query, p=2, dim=1)
        query_np = query_norm.cpu().numpy().astype('float32')
        
        # 检索
        distances, indices = self.index.search(query_np, top_k)
        
        return distances, indices
    
    @property
    def size(self) -> int:
        """获取索引大小"""
        return self.index.ntotal if self.index else 0