#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ResNet + GeM 训练脚本
使用预生成的Diamond Crops进行SimCLR对比学习
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, models

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger('train_resnet_gem')


# ==================== 数据集 ====================
class GrayscaleWithCropsDataset(Dataset):
    """
    使用预生成crops的数据集
    - 原始图像: source_tiles/gray/
    - 预生成crops: crops_output/
    """
    def __init__(self, 
                 gray_dir='./source_tiles/gray',
                 crops_dir='./crops_output',
                 size=256,
                 max_crops_per_image=20):
        
        self.gray_dir = Path(gray_dir)
        self.crops_dir = Path(crops_dir)
        self.size = size
        self.max_crops = max_crops_per_image
        
        # 获取所有有对应crops的图像
        self.valid_images = []
        self.image_to_crops = {}
        
        # 扫描crops目录
        for crop_subdir in sorted(self.crops_dir.iterdir()):
            if crop_subdir.is_dir():
                # crop_subdir的名称应该对应原始图像的stem
                img_name = crop_subdir.name
                
                # 查找对应的原始图像
                possible_extensions = ['.jpg', '.png', '.jpeg']
                original_img = None
                for ext in possible_extensions:
                    img_path = self.gray_dir / f"{img_name}{ext}"
                    if img_path.exists():
                        original_img = img_path
                        break
                
                if original_img:
                    # 获取该目录下的所有crops
                    crop_files = sorted(crop_subdir.glob('*.png'))
                    if len(crop_files) > 0:
                        self.valid_images.append(original_img)
                        self.image_to_crops[str(original_img)] = crop_files[:self.max_crops]
        
        logger.info(f"数据集初始化完成:")
        logger.info(f"  原始图像目录: {self.gray_dir}")
        logger.info(f"  Crops目录: {self.crops_dir}")
        logger.info(f"  找到有效图像-crops对: {len(self.valid_images)}")
        
        if len(self.valid_images) == 0:
            logger.warning("未找到有效的图像-crops对！请先运行prepare_crops.py生成crops")
        
        # 图像变换（用于crops）
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        img_path = self.valid_images[idx]
        crop_files = self.image_to_crops[str(img_path)]
        
        # 加载所有crops
        crops = []
        for crop_file in crop_files:
            # 加载crop（已经是灰度转RGB的）
            crop_img = Image.open(crop_file).convert('RGB')
            crop_tensor = self.transform(crop_img)
            crops.append(crop_tensor)
        
        # 填充到固定数量
        while len(crops) < self.max_crops:
            if len(crops) > 0:
                # 复制第一个crop
                crops.append(crops[0])
            else:
                # 创建空白tensor
                crops.append(torch.zeros(3, self.size, self.size))
        
        # 堆叠成单个tensor: [num_crops, 3, H, W]
        crops_tensor = torch.stack(crops[:self.max_crops], dim=0)
        
        return crops_tensor, img_path.stem


# ==================== 模型 ====================
class GeM(nn.Module):
    """Generalized Mean Pooling（可学习参数）"""
    def __init__(self, p=3.0, eps=1e-6, learn_p=True):
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = torch.tensor([p])
        self.eps = eps
        
    def forward(self, x):
        # x: B x C x H x W
        p = torch.clamp(self.p, min=self.eps, max=10.0)  # 限制最大值
        x = torch.clamp(x, min=self.eps)
        x = x.pow(p)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.pow(1.0 / p)
        return x.view(x.size(0), -1)  # B x C


class SimplifiedRetrievalModel(nn.Module):
    """
    简化的检索模型（无projection head）
    直接输出2048维特征
    """
    def __init__(self, backbone_name='resnet101', learn_gem=True, pretrained=True):
        super().__init__()
        
        # 加载backbone
        logger.info(f"加载backbone: {backbone_name}, pretrained={pretrained}")
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        
        # 移除最后的池化和全连接层
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # 冻结backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone已冻结，只训练GeM参数")
        
        # GeM池化（唯一可训练的部分）
        self.pool = GeM(p=3.0, learn_p=learn_gem)
        
        # 输出维度固定为2048（ResNet101）
        self.dim = 2048
        
    def forward(self, x):
        # 提取特征（无梯度用于backbone）
        with torch.no_grad():
            f = self.backbone(x)  # B x 2048 x H x W
        
        # 池化特征（可训练）
        g = self.pool(f)  # B x 2048
        
        # L2归一化
        z = F.normalize(g, p=2, dim=1)
        
        return z


# ==================== 损失函数 ====================
def multi_view_nt_xent_loss(views, temperature=0.1):
    """
    NT-Xent损失用于多视图（20个crops每张图像）
    Args:
        views: tensor列表，每个 [B, D]，D=2048
        temperature: 温度参数
    """
    device = views[0].device
    B = views[0].size(0)  # batch size
    N = len(views)  # 视图数量（20）
    
    # 堆叠所有视图: [B*N, D]
    all_features = torch.cat(views, dim=0)
    
    # 计算相似度矩阵
    sim_matrix = torch.matmul(all_features, all_features.t()) / temperature
    
    # 创建正样本对的标签
    labels = torch.arange(B, device=device).repeat(N)
    
    # 自相似性的mask
    mask = torch.eye(B * N, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(mask, -1e9)
    
    # 计算损失
    loss_list = []
    for i in range(B * N):
        img_idx = labels[i]
        
        # 正样本和负样本的mask
        pos_mask = (labels == img_idx) & ~mask[i]
        
        if pos_mask.sum() == 0:
            continue
        
        # 正样本相似度
        pos_sim = sim_matrix[i][pos_mask]
        
        # 所有相似度（用于分母）
        all_sim = sim_matrix[i][~mask[i]]
        
        # InfoNCE损失
        numerator = torch.logsumexp(pos_sim, dim=0)
        denominator = torch.logsumexp(all_sim, dim=0)
        
        loss_i = -numerator + denominator
        loss_list.append(loss_i)
    
    if len(loss_list) == 0:
        return torch.tensor(0.0, device=device)
    
    return torch.stack(loss_list).mean()


# ==================== 训练函数 ====================
def train_one_epoch(model, dataloader, optimizer, device, epoch, temperature=0.1):
    """训练一个epoch"""
    model.train()
    model.backbone.eval()  # backbone保持eval模式
    
    total_loss = 0.0
    p_values = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (crops_batch, img_names) in enumerate(pbar):
        # crops_batch shape: [B, 20, 3, H, W]
        B = crops_batch.size(0)
        
        # 处理所有20个视图
        batch_views = []
        
        for view_idx in range(20):
            # 获取所有图像的这个视图: [B, 3, H, W]
            view_batch = crops_batch[:, view_idx, :, :, :]
            view_batch = view_batch.to(device, non_blocking=True)
            
            # 提取特征
            features = model(view_batch)  # B x 2048
            batch_views.append(features)
        
        # 计算损失
        loss = multi_view_nt_xent_loss(batch_views, temperature=temperature)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 记录
        total_loss += loss.item()
        current_p = model.pool.p.item()
        p_values.append(current_p)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'GeM_p': f'{current_p:.3f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_p = np.mean(p_values)
    
    return avg_loss, avg_p, p_values


def validate(model, dataloader, device, temperature=0.1):
    """验证函数"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for crops_batch, _ in tqdm(dataloader, desc='Validation'):
            B = crops_batch.size(0)
            batch_views = []
            
            for view_idx in range(20):
                view_batch = crops_batch[:, view_idx, :, :, :].to(device)
                features = model(view_batch)
                batch_views.append(features)
            
            loss = multi_view_nt_xent_loss(batch_views, temperature=temperature)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def plot_training_history(history, save_dir):
    """生成训练曲线图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 损失曲线
    axes[0].plot(history['train_losses'], 'b-', linewidth=2, label='Train')
    if 'val_losses' in history and len(history['val_losses']) > 0:
        axes[0].plot(history['val_losses'], 'r--', linewidth=2, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练损失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # GeM p值演化
    axes[1].plot(history['p_values'], 'g-', linewidth=2)
    axes[1].axhline(y=3.0, color='r', linestyle='--', alpha=0.5, label='初始值 p=3.0')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('GeM p')
    axes[1].set_title('GeM参数演化')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 学习率调度
    axes[2].plot(history['learning_rates'], 'orange', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('学习率调度')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('训练进度', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'training_plots.png', dpi=150, bbox_inches='tight')
    logger.info(f"训练曲线已保存到 {save_dir / 'training_plots.png'}")


def main():
    parser = argparse.ArgumentParser(description='训练ResNet+GeM检索模型')
    
    # 数据参数
    parser.add_argument('--gray_dir', type=str, default='./source_tiles/gray',
                       help='原始灰度图像目录')
    parser.add_argument('--crops_dir', type=str, default='./crops_output',
                       help='预生成crops目录')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_resnet_gem',
                       help='模型保存目录')
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='resnet101',
                       choices=['resnet50', 'resnet101', 'resnet152'],
                       help='ResNet backbone类型')
    parser.add_argument('--learn_gem', action='store_true', default=True,
                       help='是否学习GeM参数p')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批大小（由于20个views，需要较小）')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练epoch数')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='对比学习温度参数')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建数据集
    logger.info("加载数据集...")
    dataset = GrayscaleWithCropsDataset(
        gray_dir=args.gray_dir,
        crops_dir=args.crops_dir,
        size=256,
        max_crops_per_image=20
    )
    
    if len(dataset) == 0:
        logger.error("错误: 未找到有效的图像-crop对！")
        logger.error("请先运行: python scripts/prepare_crops.py")
        return
    
    # 划分训练集和验证集
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    logger.info(f"数据集划分: 训练集 {train_size}, 验证集 {val_size}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    logger.info("初始化模型...")
    model = SimplifiedRetrievalModel(
        backbone_name=args.backbone,
        learn_gem=args.learn_gem,
        pretrained=True
    )
    model = model.to(device)
    
    # 只优化GeM池化参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"可训练参数数量: {len(trainable_params)}")
    logger.info(f"总参数量: {sum(p.numel() for p in trainable_params)}")
    
    # 优化器（只针对GeM的p参数）
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 训练历史
    history = {
        'train_losses': [],
        'val_losses': [],
        'p_values': [],
        'learning_rates': []
    }
    
    # 打印训练配置
    logger.info("\n" + "="*50)
    logger.info("训练配置:")
    logger.info(f"  数据集: {args.gray_dir}")
    logger.info(f"  Crops: {args.crops_dir}")
    logger.info(f"  有效对数: {len(dataset)}")
    logger.info(f"  批次: {args.batch_size}")
    logger.info(f"  总epochs: {args.epochs}")
    logger.info(f"  初始学习率: {args.lr}")
    logger.info(f"  温度参数: {args.temperature}")
    logger.info(f"  设备: {device}")
    logger.info(f"  输出维度: 2048")
    logger.info(f"  初始GeM p: {model.pool.p.item():.3f}")
    logger.info("="*50 + "\n")
    
    best_loss = float('inf')
    
    # 训练循环
    for epoch in range(1, args.epochs + 1):
        # 训练
        avg_loss, avg_p, p_values = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.temperature
        )
        
        # 验证
        val_loss = validate(model, val_loader, device, args.temperature)
        
        # 更新学习率
        scheduler.step()
        
        # 保存历史
        history['train_losses'].append(avg_loss)
        history['val_losses'].append(val_loss)
        history['p_values'].append(avg_p)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # 打印摘要
        logger.info(f"\nEpoch {epoch}/{args.epochs} 摘要:")
        logger.info(f"  训练损失: {avg_loss:.4f}")
        logger.info(f"  验证损失: {val_loss:.4f}")
        logger.info(f"  GeM p: {avg_p:.4f} (std: {np.std(p_values):.4f})")
        logger.info(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'gem_p': model.pool.p.item(),
            'history': history,
            'config': vars(args)
        }
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, save_dir / 'best_model.pth')
            logger.info(f"  ✓ 新的最佳模型已保存 (val_loss: {val_loss:.4f})")
        
        # 定期保存检查点
        if epoch % 5 == 0:
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # 保存最终模型
    torch.save(checkpoint, save_dir / 'final_model.pth')
    
    # 保存训练历史
    with open(save_dir / 'history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    # 绘制训练曲线
    plot_training_history(history, save_dir)
    
    logger.info(f"\n{'='*50}")
    logger.info("训练完成！")
    logger.info(f"最佳验证损失: {best_loss:.4f}")
    logger.info(f"最终GeM p: {model.pool.p.item():.4f}")
    logger.info(f"检查点保存在: {save_dir}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()