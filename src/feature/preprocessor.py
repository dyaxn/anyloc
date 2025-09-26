"""
图像预处理模块
统一处理数据库图像和查询图像的预处理
"""
from pathlib import Path
from typing import Tuple
import torch
import torchvision.transforms as T
from PIL import Image


class ImagePreprocessor:
    """统一的图像预处理器"""
    
    def __init__(self, 
                 target_size: int = 896,
                 patch_size: int = 14,
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        """
        初始化预处理器
        
        Args:
            target_size: 目标图像大小
            patch_size: ViT的patch大小
            mean: 归一化均值
            std: 归一化标准差
        """
        self.target_size = target_size
        self.patch_size = patch_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    
    def preprocess(self, image_path: str, is_query: bool = False) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image_path: 图像路径
            is_query: 是否为查询图像（查询图像保留黑边）
        
        Returns:
            预处理后的张量
        """
        img = self._load_image(image_path)
        
        if is_query:
            # 查询图像：保留黑边，直接resize到14的倍数
            img = self._resize_to_patches(img, keep_aspect=True)
        else:
            # 数据库图像：略微放大到14的倍数
            img = self._resize_database_image(img)
        
        return self.transform(img)
    
    def _load_image(self, image_path: str) -> Image.Image:
        """加载并转换图像为RGB"""
        img = Image.open(image_path)
        if img.mode != 'RGB':
            if img.mode != 'L':
                img = img.convert('L')
            img = img.convert('RGB')
        return img
    
    def _resize_to_patches(self, img: Image.Image, keep_aspect: bool = True) -> Image.Image:
        """调整图像大小到14的倍数"""
        w, h = img.size
        
        if keep_aspect and max(w, h) > self.target_size:
            # 保持宽高比缩放
            scale = self.target_size / max(w, h)
            w = int(w * scale)
            h = int(h * scale)
        
        # 调整到patch_size的倍数
        target_w = (w // self.patch_size) * self.patch_size
        target_h = (h // self.patch_size) * self.patch_size
        
        # 确保不会太小
        target_w = max(self.patch_size, target_w)
        target_h = max(self.patch_size, target_h)
        
        return img.resize((target_w, target_h), Image.LANCZOS)
    
    def _resize_database_image(self, img: Image.Image) -> Image.Image:
        """数据库图像resize（向上取整到14的倍数）"""
        w, h = img.size
        target_w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size
        target_h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
        return img.resize((target_w, target_h), Image.LANCZOS)