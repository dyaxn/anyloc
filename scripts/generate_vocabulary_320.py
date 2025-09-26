#!/usr/bin/env python3
"""
生成自定义VLAD词汇表（使用322×322分辨率）
基于AnyLoc的内存优化策略
"""

import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import yaml
import random
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import gc
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.feature.extractor import DinoV2Extractor


class VocabularyGenerator:
    """词汇表生成器"""
    
    def __init__(self, 
                 model_name: str = "dinov2_vitl14",
                 layer: int = 23,
                 facet: str = "value",
                 device: str = "cuda",
                 resize_dim: int = 322):
        """
        初始化生成器
        
        Args:
            model_name: DINOv2模型名称
            layer: 提取层
            facet: 特征分支 
            device: 计算设备
            resize_dim: 图像resize尺寸（322是14的倍数）
        """
        self.device = device
        
        # 确保是14的倍数
        self.resize_dim = (resize_dim // 14) * 14
        if self.resize_dim != resize_dim:
            print(f"⚠️ Adjusted resize from {resize_dim} to {self.resize_dim} (multiple of 14)")
        
        self.patch_size = 14
        
        # 计算patches数量
        self.n_patches = (self.resize_dim // self.patch_size) ** 2
        print(f"Image size: {self.resize_dim}×{self.resize_dim}")
        print(f"Patches per image: {self.n_patches}")
        
        # 初始化模型
        print("Loading DINOv2 model...")
        self.extractor = DinoV2Extractor(
            model_name=model_name,
            layer=layer,
            device=device
        )
        
        # 图像预处理
        self.transform = T.Compose([
            T.Resize((self.resize_dim, self.resize_dim)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features_batch(self, image_paths: list, batch_size: int = 16):
        """
        批量提取图像特征（带详细进度）
        
        Args:
            image_paths: 图像路径列表
            batch_size: 批处理大小
            
        Returns:
            所有图像的局部特征 [N_total_patches, D]
        """
        print("\nInitializing model...")
        self.extractor.init_model()
        print("Model ready!")
        
        all_features = []
        total_images = len(image_paths)
        total_batches = (total_images + batch_size - 1) // batch_size
        
        # 主进度条
        with tqdm(total=total_images, desc="Extracting features", unit="img") as pbar:
            start_time = time.time()
            
            for batch_idx in range(0, total_images, batch_size):
                batch_end = min(batch_idx + batch_size, total_images)
                batch_paths = image_paths[batch_idx:batch_end]
                current_batch_size = len(batch_paths)
                
                # 加载图像
                batch_tensors = []
                for img_path in batch_paths:
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)
                
                # 批量推理
                batch = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    # 提取特征 [B, N_patches, D]
                    features = self.extractor.extract(batch)
                    # 展平批次维度 [B*N_patches, D]
                    features = features.reshape(-1, features.shape[-1])
                    all_features.append(features.cpu())
                
                # 更新进度条
                pbar.update(current_batch_size)
                
                # 显示额外信息
                elapsed = time.time() - start_time
                avg_speed = (batch_idx + current_batch_size) / elapsed
                eta = (total_images - batch_idx - current_batch_size) / avg_speed if avg_speed > 0 else 0
                
                pbar.set_postfix({
                    'batch': f"{batch_idx//batch_size + 1}/{total_batches}",
                    'speed': f"{avg_speed:.1f} img/s",
                    'ETA': f"{eta:.0f}s"
                })
                
                # 清理GPU内存
                if self.device == 'cuda' and (batch_idx // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
        
        # 合并所有特征
        print("\nConcatenating features...")
        all_features = torch.cat(all_features, dim=0)
        
        return all_features
    
    def cluster_features(self, features: torch.Tensor, n_clusters: int = 32):
        """
        K-means聚类（简化版，避免inertia_错误）
        
        Args:
            features: 特征张量 [N_total_patches, D]
            n_clusters: 聚类数量
            
        Returns:
            聚类中心 [K, D]
        """
        print(f"\n{'='*40}")
        print(f"Clustering {features.shape[0]:,} features into {n_clusters} clusters")
        print(f"{'='*40}")
        
        features_np = features.numpy().astype('float32')
        
        # 计算批次大小
        batch_size = min(10000, features.shape[0])
        print(f"Batch size: {batch_size}")
        print(f"Feature shape: {features_np.shape}")
        print("\nStarting K-means clustering...")
        print("This may take a few minutes...\n")
        
        # 创建MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            max_iter=100,
            n_init=3,
            random_state=42,
            verbose=1,  # 开启内置进度显示
            compute_labels=False  # 不计算标签，节省内存
        )
        
        # 直接fit（会显示内置进度）
        kmeans.fit(features_np)
        
        # 获取聚类中心
        centers = torch.from_numpy(kmeans.cluster_centers_)
        
        print(f"\n✓ Clustering complete!")
        print(f"  Centers shape: {centers.shape}")
        print(f"  Final inertia: {kmeans.inertia_:.2f}")
        
        return centers


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate VLAD vocabulary")
    parser.add_argument('--data_dir', type=str, 
                       default='/root/autodl-tmp/data/source_tiles/img',
                       help='Database images directory')
    parser.add_argument('--n_clusters', type=int, default=32,
                       help='Number of VLAD clusters')
    parser.add_argument('--max_images', type=int, default=6800,
                       help='Maximum images to use')
    parser.add_argument('--subsample', type=int, default=1,
                       help='Subsample factor (1=use all, 2=use half, etc.)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for feature extraction')
    parser.add_argument('--output', type=str, 
                       default='models/custom_322/c_centers.pt',
                       help='Output vocabulary file')
    parser.add_argument('--resize', type=int, default=322,
                       help='Resize dimension (322 for clustering, 896 for retrieval)')
    args = parser.parse_args()
    
    print("="*60)
    print("VLAD Vocabulary Generation (Memory Optimized)")
    print("="*60)
    
    # 获取图像列表
    print("\n📁 Scanning image directory...")
    image_dir = Path(args.data_dir)
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    
    image_paths = []
    for ext in image_extensions:
        found = list(image_dir.glob(ext))
        image_paths.extend(found)
    
    image_paths = sorted(image_paths)[:args.max_images]
    print(f"✓ Found {len(image_paths)} images")
    
    # 降采样
    if args.subsample > 1:
        print(f"\n📉 Subsampling with factor {args.subsample}")
        image_paths = image_paths[::args.subsample]
        print(f"✓ After subsampling: {len(image_paths)} images")
    
    # 限制最大图像数
    if len(image_paths) > args.max_images:
        print(f"\n🎲 Limiting to {args.max_images} images")
        image_paths = image_paths[:args.max_images]
    
    print(f"\n📊 Final dataset: {len(image_paths)} images")
    
    # 内存估算
    print("\n💾 Memory estimation:")
    resize_actual = (args.resize // 14) * 14
    patches_per_img = (resize_actual // 14) ** 2
    mem_per_img = patches_per_img * 1536 * 4 / (1024**2)  # MB
    total_mem = mem_per_img * len(image_paths) / 1024  # GB
    print(f"  Resize dimension: {resize_actual}×{resize_actual}")
    print(f"  Patches per image: {patches_per_img}")
    print(f"  Memory per image: {mem_per_img:.2f} MB")
    print(f"  Total estimated: {total_mem:.2f} GB")
    
    if total_mem > 50:
        print(f"⚠️  Warning: Estimated memory usage is high ({total_mem:.1f}GB)")
        print("   Consider using --subsample or reducing --max_images")
    
    # 初始化生成器
    print("\n🚀 Initializing generator...")
    generator = VocabularyGenerator(
        model_name="dinov2_vitl14",
        layer=23,
        facet="value",
        device="cuda" if torch.cuda.is_available() else "cpu",
        resize_dim=args.resize
    )
    print("✓ Generator ready")
    
    # 提取特征
    print("\n" + "="*40)
    print("Step 1/3: Feature Extraction")
    print("="*40)
    
    start_time = time.time()
    features = generator.extract_features_batch(
        image_paths, 
        batch_size=args.batch_size
    )
    
    extraction_time = time.time() - start_time
    print(f"\n✓ Feature extraction complete!")
    print(f"  Shape: {features.shape}")
    print(f"  Memory: {features.nbytes / (1024**3):.2f} GB")
    print(f"  Time: {extraction_time:.1f} seconds")
    print(f"  Speed: {len(image_paths)/extraction_time:.1f} img/s")
    
    # K-means聚类
    print("\n" + "="*40)
    print("Step 2/3: K-means Clustering")
    print("="*40)
    
    start_time = time.time()
    centers = generator.cluster_features(features, n_clusters=args.n_clusters)
    clustering_time = time.time() - start_time
    
    print(f"  Time: {clustering_time:.1f} seconds")
    
    # 清理内存
    print("\n🧹 Cleaning memory...")
    del features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ Memory cleaned")
    
    # 保存词汇表
    print("\n" + "="*40)
    print("Step 3/3: Saving Vocabulary")
    print("="*40)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存格式与AnyLoc一致
    vocab_data = {
        'c_centers': centers,
        'n_clusters': args.n_clusters,
        'feature_dim': centers.shape[1],
        'n_training_images': len(image_paths),
        'resize_dim': generator.resize_dim,  # 使用实际的resize维度
        'source_dir': str(args.data_dir),
        'config': {
            'model': 'dinov2_vitl14',
            'layer': 23,
            'facet': 'value'
        }
    }
    
    print("💾 Saving vocabulary...")
    torch.save(vocab_data, output_path)
    print(f"✅ Vocabulary saved to: {output_path}")
    
    # 显示摘要
    print("\n" + "="*60)
    print("📋 Summary")
    print("="*60)
    print(f"  Clusters: {args.n_clusters}")
    print(f"  Feature dimension: {centers.shape[1]}")
    print(f"  Training images: {len(image_paths)}")
    print(f"  Total time: {extraction_time + clustering_time:.1f} seconds")
    
    # 验证加载
    print("\n🔍 Verifying saved file...")
    loaded = torch.load(output_path)
    print(f"✓ Verification successful - centers shape: {loaded['c_centers'].shape}")


if __name__ == "__main__":
    main()