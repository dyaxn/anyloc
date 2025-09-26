#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLAD词汇表训练脚本
支持DINOv2和DINOv3模型的词汇表训练
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import time
import json
from torchvision import transforms as tvf

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.feature.extractor import FeatureExtractor
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger('train_vocab')

# 尝试导入fast_pytorch_kmeans
try:
    import fast_pytorch_kmeans as fpk
    HAS_FAST_KMEANS = True
    logger.info("使用fast_pytorch_kmeans进行GPU加速聚类")
except ImportError:
    HAS_FAST_KMEANS = False
    logger.warning("未找到fast_pytorch_kmeans，将使用sklearn的CPU聚类")
    logger.warning("安装: pip install fast-pytorch-kmeans")
    from sklearn.cluster import KMeans


class VocabularyTrainer:
    """VLAD词汇表训练器"""
    
    def __init__(self, 
                 model_type: str = "dinov2_vitg14",
                 layer: int = 31,
                 facet: str = "value",
                 num_clusters: int = 64,
                 device: str = "cuda"):
        """
        初始化词汇表训练器
        
        Args:
            model_type: 模型类型 (dinov2_vitg14, dinov3_vitb16等)
            layer: 提取特征的层
            facet: 特征类型 (query, key, value)
            num_clusters: 聚类中心数量
            device: 设备
        """
        self.model_type = model_type
        self.layer = layer
        self.facet = facet
        self.num_clusters = num_clusters
        self.device = device
        
        # 初始化特征提取器
        logger.info(f"初始化特征提取器: {model_type}")
        self.extractor = FeatureExtractor(
            model_type=model_type,
            layer=layer,
            facet=facet,
            device=device
        )
        
        # 获取模型配置
        self.patch_size = self.extractor.get_patch_size()
        self.feat_dim = self.extractor.get_feature_dim()
        
        logger.info(f"模型配置:")
        logger.info(f"  Patch大小: {self.patch_size}")
        logger.info(f"  特征维度: {self.feat_dim}")
        logger.info(f"  聚类数量: {num_clusters}")
        
    def prepare_dataset(self, data_dir: Path, 
                        target_size: int = 320,
                        sub_sample: int = 1,
                        max_images: int = None):
        """
        准备数据集
        
        Args:
            data_dir: 数据目录
            target_size: 目标图像尺寸（正方形）
            sub_sample: 降采样率（每N张图取1张）
            max_images: 最大图像数
            
        Returns:
            image_paths: 图像路径列表
            transform: 图像变换
        """
        # 获取所有图像
        image_paths = []
        extensions = ['*.jpg', '*.jpeg', '*.png']
        
        for ext in extensions:
            image_paths.extend(list(data_dir.glob(ext)))
        
        # 排序确保一致性
        image_paths = sorted(image_paths)
        
        # 降采样
        if sub_sample > 1:
            image_paths = image_paths[::sub_sample]
            logger.info(f"降采样: 每{sub_sample}张取1张")
        
        # 限制数量
        if max_images:
            image_paths = image_paths[:max_images]
            logger.info(f"限制图像数量: {max_images}")
        
        logger.info(f"数据集准备完成: {len(image_paths)}张图像")
        
        # 创建变换（注意：先归一化再resize，遵循官方流程）
        transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
            tvf.Resize((target_size, target_size))
        ])
        
        return image_paths, transform
    
    def extract_patch_descriptors(self, image_paths: list, transform):
        """
        提取所有图像的patch描述符
        
        Args:
            image_paths: 图像路径列表
            transform: 图像变换
            
        Returns:
            all_descriptors: 所有patch描述符 [N_total_patches, D]
        """
        all_descriptors = []
        patches_per_img = []
        
        # 计算预期的patches数量
        if self.patch_size:
            # DINO模型
            patches_per_side = 320 // self.patch_size
            expected_patches = patches_per_side ** 2
            logger.info(f"每张图像预期patches: {expected_patches} ({patches_per_side}×{patches_per_side})")
        
        logger.info("开始提取特征...")
        
        for img_path in tqdm(image_paths, desc="提取特征"):
            # 加载和预处理图像
            img = Image.open(img_path).convert('RGB')
            
            # 对数据库图像，先转灰度再转回RGB（模拟实际使用）
            img = img.convert('L').convert('RGB')
            
            # 应用变换
            img_tensor = transform(img)
            
            # 确保是正确的尺寸
            c, h, w = img_tensor.shape
            
            # 调整到patch_size的倍数
            if self.patch_size:
                h_new = (h // self.patch_size) * self.patch_size
                w_new = (w // self.patch_size) * self.patch_size
                
                if h_new != h or w_new != w:
                    img_tensor = tvf.CenterCrop((h_new, w_new))(img_tensor)
            
            # 添加batch维度
            img_batch = img_tensor.unsqueeze(0).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                features = self.extractor.extract(img_batch)  # [1, N_patches, D]
                
                # 立即移到CPU节省GPU内存
                features = features.cpu()
                
                # 记录patches数量
                patches_per_img.append(features.shape[1])
                
                # 添加到列表
                all_descriptors.append(features[0])  # [N_patches, D]
            
            # 定期清理GPU缓存
            if len(all_descriptors) % 100 == 0:
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        # 合并所有描述符
        logger.info("合并所有patch描述符...")
        all_descriptors = torch.cat(all_descriptors, dim=0)  # [N_total_patches, D]
        
        # 统计信息
        total_patches = all_descriptors.shape[0]
        avg_patches = np.mean(patches_per_img)
        
        logger.info(f"特征提取完成:")
        logger.info(f"  总patches数: {total_patches:,}")
        logger.info(f"  平均每张图: {avg_patches:.1f} patches")
        logger.info(f"  描述符形状: {all_descriptors.shape}")
        logger.info(f"  预计内存需求: ~{total_patches * self.feat_dim * 4 / 1024**3:.1f}GB")
        
        return all_descriptors
    
    def train_vocabulary(self, descriptors: torch.Tensor, max_iter: int = 100):
        """
        训练VLAD词汇表（聚类中心）
        
        Args:
            descriptors: 所有patch描述符 [N, D]
            max_iter: 最大迭代次数
            
        Returns:
            centers: 聚类中心 [K, D]
        """
        logger.info(f"开始训练词汇表...")
        logger.info(f"  输入描述符: {descriptors.shape}")
        logger.info(f"  聚类数量: {self.num_clusters}")
        logger.info(f"  最大迭代: {max_iter}")
        
        # L2归一化
        descriptors = F.normalize(descriptors, p=2, dim=1)
        
        start_time = time.time()
        
        if HAS_FAST_KMEANS and self.device == 'cuda':
            # 使用GPU加速的K-means
            logger.info("使用GPU加速K-means聚类...")
            
            # 将描述符移到GPU
            descriptors_gpu = descriptors.to(self.device)
            
            # 使用fast_pytorch_kmeans
            kmeans = fpk.KMeans(
                n_clusters=self.num_clusters,
                mode='cosine',  # 使用余弦相似度
                verbose=1,
                max_iter=max_iter
            )
            
            # 训练
            kmeans.fit(descriptors_gpu)
            
            # 获取聚类中心
            centers = kmeans.centroids.cpu()
            
        else:
            # 使用CPU的sklearn K-means
            logger.info("使用CPU K-means聚类（较慢）...")
            
            # 转换为numpy
            descriptors_np = descriptors.numpy()
            
            # 使用sklearn
            kmeans = KMeans(
                n_clusters=self.num_clusters,
                max_iter=max_iter,
                n_init=10,
                verbose=1
            )
            
            # 训练
            kmeans.fit(descriptors_np)
            
            # 获取聚类中心
            centers = torch.from_numpy(kmeans.cluster_centers_).float()
        
        elapsed_time = time.time() - start_time
        logger.info(f"聚类完成，耗时: {elapsed_time:.1f}秒")
        logger.info(f"聚类中心形状: {centers.shape}")
        
        return centers
    
    def save_vocabulary(self, centers: torch.Tensor, save_path: Path, metadata: dict = None):
        """
        保存词汇表
        
        Args:
            centers: 聚类中心
            save_path: 保存路径
            metadata: 额外的元数据
        """
        # 准备保存数据
        save_data = {
            'c_centers': centers,  # 兼容旧格式
            'centers': centers,    # 新格式
            'num_clusters': self.num_clusters,
            'feat_dim': self.feat_dim,
            'config': {
                'model': self.model_type,
                'layer': self.layer,
                'facet': self.facet,
                'num_clusters': self.num_clusters,
                'patch_size': self.patch_size,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # 添加额外元数据
        if metadata:
            save_data['training_info'] = metadata
        
        # 保存
        torch.save(save_data, save_path)
        
        # 同时保存JSON元数据
        json_path = save_path.parent / f"{save_path.stem}_info.json"
        with open(json_path, 'w') as f:
            json_info = {
                'model': self.model_type,
                'layer': self.layer,
                'facet': self.facet,
                'num_clusters': self.num_clusters,
                'patch_size': self.patch_size,
                'feat_dim': self.feat_dim,
                'file_size_mb': save_path.stat().st_size / 1024 / 1024,
                'timestamp': datetime.now().isoformat()
            }
            if metadata:
                json_info['training_info'] = {k: str(v) for k, v in metadata.items()}
            
            json.dump(json_info, f, indent=2)
        
        logger.info(f"词汇表已保存:")
        logger.info(f"  文件: {save_path}")
        logger.info(f"  大小: {save_path.stat().st_size / 1024 / 1024:.2f} MB")
        logger.info(f"  元数据: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='训练VLAD词汇表')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='训练数据目录')
    parser.add_argument('--output_dir', type=str, default='./models/vocabularies',
                       help='输出目录')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='dinov2_vitg14',
                       help='模型类型 (dinov2_vitg14, dinov3_vitb16等)')
    parser.add_argument('--layer', type=int, default=None,
                       help='提取层 (默认: DINOv2=31, DINOv3=11)')
    parser.add_argument('--facet', type=str, default='value',
                       choices=['query', 'key', 'value', 'token'],
                       help='特征类型')
    
    # 聚类参数
    parser.add_argument('--num_clusters', type=int, default=64,
                       help='聚类中心数量')
    parser.add_argument('--max_iter', type=int, default=100,
                       help='K-means最大迭代次数')
    
    # 数据集参数
    parser.add_argument('--image_size', type=int, default=320,
                       help='图像尺寸（正方形）')
    parser.add_argument('--sub_sample', type=int, default=1,
                       help='降采样率（1=全部，2=一半）')
    parser.add_argument('--max_images', type=int, default=None,
                       help='最大图像数量')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 自动设置layer默认值
    if args.layer is None:
        if 'dinov3' in args.model:
            args.layer = 11
        else:
            args.layer = 31
    
    logger.info("="*60)
    logger.info("VLAD词汇表训练")
    logger.info("="*60)
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"模型: {args.model}")
    logger.info(f"层: {args.layer}")
    logger.info(f"特征: {args.facet}")
    logger.info(f"聚类数: {args.num_clusters}")
    logger.info("="*60)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化训练器
    trainer = VocabularyTrainer(
        model_type=args.model,
        layer=args.layer,
        facet=args.facet,
        num_clusters=args.num_clusters,
        device=args.device
    )
    
    # 准备数据集
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return
    
    image_paths, transform = trainer.prepare_dataset(
        data_dir,
        target_size=args.image_size,
        sub_sample=args.sub_sample,
        max_images=args.max_images
    )
    
    if len(image_paths) == 0:
        logger.error("未找到图像文件！")
        return
    
    # 提取特征
    descriptors = trainer.extract_patch_descriptors(image_paths, transform)
    
    # 训练词汇表
    centers = trainer.train_vocabulary(descriptors, max_iter=args.max_iter)
    
    # 生成输出文件名
    model_name = args.model.replace('/', '_')
    output_file = output_dir / f"{model_name}_vlad_{args.num_clusters}.pt"
    
    # 保存词汇表
    metadata = {
        'data_dir': str(data_dir),
        'num_images': len(image_paths),
        'sub_sample': args.sub_sample,
        'image_size': args.image_size,
        'total_patches': descriptors.shape[0]
    }
    
    trainer.save_vocabulary(centers, output_file, metadata)
    
    logger.info("\n" + "="*60)
    logger.info("训练完成！")
    logger.info(f"词汇表保存在: {output_file}")
    logger.info("="*60)
    
    # 使用提示
    logger.info("\n使用新词汇表进行检索:")
    logger.info(f"python scripts/run_retrieval.py --model {args.model} --agg vlad --vocab {output_file}")


if __name__ == "__main__":
    main()