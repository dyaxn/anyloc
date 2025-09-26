#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Image Preprocessor
支持DINOv2 (14x) 和 DINOv3 (16x) 的自适应预处理
"""

import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    增强版图像预处理器
    自动适配不同模型的patch size要求
    """
    
    def __init__(self, 
                 target_size: Optional[Union[int, Tuple[int, int]]] = None,
                 mode: str = 'database',
                 patch_size: Optional[int] = None):
        """
        初始化预处理器
        
        Args:
            target_size: 目标尺寸（可选，None表示自动计算）
            mode: 处理模式 ('database' 或 'query')
            patch_size: 模型的patch大小（14 for DINOv2, 16 for DINOv3, None for ResNet）
        """
        self.mode = mode
        self.patch_size = patch_size
        
        # 设置默认尺寸
        if target_size is None:
            # 根据模式和patch_size自动确定
            if mode == 'database':
                self.base_size = (720, 540)  # 数据库图像原始尺寸
            else:
                self.base_size = (900, 900)  # 查询图像原始尺寸
        elif isinstance(target_size, int):
            self.base_size = (target_size, target_size)
        else:
            self.base_size = target_size
            
        # 计算实际目标尺寸（调整到patch_size的倍数）
        self.target_size = self._compute_target_size(self.base_size)
        
        # 创建标准化变换
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        logger.info(f"预处理器初始化:")
        logger.info(f"  模式: {mode}")
        logger.info(f"  Patch大小: {patch_size}")
        logger.info(f"  基础尺寸: {self.base_size}")
        logger.info(f"  目标尺寸: {self.target_size}")
        
    def _compute_target_size(self, base_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        计算调整到patch_size倍数的目标尺寸
        
        Args:
            base_size: 基础尺寸 (width, height)
            
        Returns:
            目标尺寸 (width, height)
        """
        if self.patch_size is None:
            # ResNet等CNN模型，不需要特殊调整
            return base_size
            
        width, height = base_size
        
        if self.mode == 'database':
            # 数据库图像：向上取整到patch_size的倍数
            target_width = ((width + self.patch_size - 1) // self.patch_size) * self.patch_size
            target_height = ((height + self.patch_size - 1) // self.patch_size) * self.patch_size
        else:
            # 查询图像：向下取整到patch_size的倍数（保留黑边）
            target_width = (width // self.patch_size) * self.patch_size
            target_height = (height // self.patch_size) * self.patch_size
            
            # 确保不为0
            if target_width == 0:
                target_width = self.patch_size
            if target_height == 0:
                target_height = self.patch_size
                
        return (target_width, target_height)
    
    def process(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        处理图像文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            处理后的图像张量 [3, H, W]
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 加载图像
        image = Image.open(image_path)
        
        # 处理不同的图像模式
        if image.mode == 'L':
            # 灰度图像转RGB
            image = image.convert('RGB')
        elif image.mode == 'RGBA':
            # RGBA转RGB（移除alpha通道）
            image = image.convert('RGB')
        elif image.mode != 'RGB':
            # 其他模式都转为RGB
            image = image.convert('RGB')
        
        # 应用预处理
        return self.process_pil(image)
    
    def process_pil(self, image: Image.Image) -> torch.Tensor:
        """
        处理PIL图像
        
        Args:
            image: PIL图像对象
            
        Returns:
            处理后的图像张量 [3, H, W]
        """
        # 根据模式进行不同的预处理
        if self.mode == 'database':
            # 数据库图像：彩色→灰度→RGB
            if image.mode != 'L':
                image = image.convert('L')
            image = image.convert('RGB')
        else:
            # 查询图像：保持原样（通常已经是灰度）
            if image.mode == 'L':
                image = image.convert('RGB')
                
        # 调整尺寸
        image = image.resize(self.target_size, Image.LANCZOS)
        
        # 转换为tensor
        image_tensor = transforms.ToTensor()(image)
        
        # 归一化
        image_tensor = self.normalize(image_tensor)
        
        return image_tensor
    
    def process_numpy(self, image_array: np.ndarray) -> torch.Tensor:
        """
        处理numpy数组
        
        Args:
            image_array: numpy数组 (H, W, C) 或 (H, W)
            
        Returns:
            处理后的图像张量 [3, H, W]
        """
        # 转换为PIL图像
        if image_array.ndim == 2:
            # 灰度图像
            image = Image.fromarray(image_array, mode='L')
        elif image_array.ndim == 3:
            if image_array.shape[2] == 3:
                # RGB图像
                image = Image.fromarray(image_array, mode='RGB')
            elif image_array.shape[2] == 4:
                # RGBA图像
                image = Image.fromarray(image_array, mode='RGBA')
            else:
                raise ValueError(f"不支持的通道数: {image_array.shape[2]}")
        else:
            raise ValueError(f"不支持的数组维度: {image_array.ndim}")
        
        return self.process_pil(image)
    
    def batch_process(self, 
                     image_paths: list,
                     return_failed: bool = False) -> Tuple[torch.Tensor, list]:
        """
        批量处理图像
        
        Args:
            image_paths: 图像路径列表
            return_failed: 是否返回失败的路径
            
        Returns:
            (batch_tensor, failed_paths): 批量张量和失败路径列表
        """
        processed = []
        failed_paths = []
        
        for img_path in image_paths:
            try:
                img_tensor = self.process(img_path)
                processed.append(img_tensor)
            except Exception as e:
                logger.warning(f"处理图像失败 {img_path}: {e}")
                if return_failed:
                    failed_paths.append(img_path)
                    
        if len(processed) > 0:
            # 堆叠成batch
            batch_tensor = torch.stack(processed, dim=0)
        else:
            batch_tensor = torch.empty(0, 3, self.target_size[1], self.target_size[0])
            
        if return_failed:
            return batch_tensor, failed_paths
        else:
            return batch_tensor
    
    def get_output_size(self) -> Tuple[int, int]:
        """
        获取输出图像尺寸
        
        Returns:
            (width, height)
        """
        return self.target_size
    
    def get_num_patches(self) -> Optional[Tuple[int, int]]:
        """
        获取patches数量
        
        Returns:
            (patches_width, patches_height) 或 None（对于CNN模型）
        """
        if self.patch_size is None:
            return None
            
        width, height = self.target_size
        patches_w = width // self.patch_size
        patches_h = height // self.patch_size
        
        return (patches_w, patches_h)
    
    def __repr__(self):
        return (f"ImagePreprocessor(mode={self.mode}, "
                f"patch_size={self.patch_size}, "
                f"target_size={self.target_size})")


class AdaptivePreprocessor:
    """
    自适应预处理器
    根据模型类型自动选择合适的预处理策略
    """
    
    def __init__(self, model_type: str, mode: str = 'database'):
        """
        初始化自适应预处理器
        
        Args:
            model_type: 模型类型
            mode: 处理模式
        """
        self.model_type = model_type.lower()
        self.mode = mode
        
        # 确定patch size
        if 'dinov3' in self.model_type:
            self.patch_size = 16
        elif 'dinov2' in self.model_type or 'dino' in self.model_type:
            self.patch_size = 14
        else:
            # ResNet等CNN模型
            self.patch_size = None
            
        # 创建基础预处理器
        self.preprocessor = ImagePreprocessor(
            target_size=None,
            mode=mode,
            patch_size=self.patch_size
        )
        
        logger.info(f"自适应预处理器初始化:")
        logger.info(f"  模型: {model_type}")
        logger.info(f"  检测到的patch大小: {self.patch_size}")
        
    def process(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        处理图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            处理后的张量
        """
        return self.preprocessor.process(image_path)
    
    def get_config(self) -> dict:
        """
        获取预处理配置
        
        Returns:
            配置字典
        """
        return {
            'model_type': self.model_type,
            'mode': self.mode,
            'patch_size': self.patch_size,
            'target_size': self.preprocessor.target_size,
            'num_patches': self.preprocessor.get_num_patches()
        }


def create_preprocessor(config: dict) -> ImagePreprocessor:
    """
    工厂函数：根据配置创建预处理器
    
    Args:
        config: 配置字典
        
    Returns:
        预处理器实例
    """
    model_type = config.get('model', {}).get('extractor', {}).get('type', 'dinov2_vitg14')
    mode = config.get('mode', 'database')
    
    # 自动检测patch size
    if 'dinov3' in model_type.lower():
        patch_size = 16
    elif 'dino' in model_type.lower():
        patch_size = 14
    else:
        patch_size = None
    
    # 获取目标尺寸配置
    preprocessing_config = config.get('preprocessing', {})
    if mode == 'database':
        base_size = preprocessing_config.get('database', {}).get('base_size', 720)
    else:
        base_size = preprocessing_config.get('query', {}).get('base_size', 900)
    
    return ImagePreprocessor(
        target_size=base_size,
        mode=mode,
        patch_size=patch_size
    )


# 测试函数
def test_preprocessor():
    """测试预处理器功能"""
    
    # 测试DINOv2预处理（14x）
    print("测试DINOv2预处理器:")
    prep_v2 = ImagePreprocessor(patch_size=14, mode='database')
    print(f"  目标尺寸: {prep_v2.target_size}")
    print(f"  Patches数量: {prep_v2.get_num_patches()}")
    
    # 测试DINOv3预处理（16x）
    print("\n测试DINOv3预处理器:")
    prep_v3 = ImagePreprocessor(patch_size=16, mode='database')
    print(f"  目标尺寸: {prep_v3.target_size}")
    print(f"  Patches数量: {prep_v3.get_num_patches()}")
    
    # 测试ResNet预处理（无patch要求）
    print("\n测试ResNet预处理器:")
    prep_resnet = ImagePreprocessor(patch_size=None, mode='database', target_size=256)
    print(f"  目标尺寸: {prep_resnet.target_size}")
    print(f"  Patches数量: {prep_resnet.get_num_patches()}")
    
    # 测试自适应预处理器
    print("\n测试自适应预处理器:")
    models = ['dinov2_vitg14', 'dinov3_vitb16', 'resnet101']
    for model in models:
        adaptive = AdaptivePreprocessor(model)
        config = adaptive.get_config()
        print(f"  {model}: patch_size={config['patch_size']}, target={config['target_size']}")


if __name__ == "__main__":
    test_preprocessor()