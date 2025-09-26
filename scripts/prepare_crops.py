#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diamond Crops数据准备脚本
用于生成SimCLR训练所需的菱形裁剪图像
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger('prepare_crops')


class DiamondCropAugmentation:
    """菱形裁剪增强器"""
    
    def __init__(self, input_size=(720, 540), canvas_size=900):
        """
        初始化
        Args:
            input_size: 输入图像尺寸 (宽, 高)
            canvas_size: 输出画布尺寸（正方形）
        """
        self.input_w, self.input_h = input_size
        self.canvas_size = canvas_size
        
        # 计算内接菱形参数
        self.center = canvas_size // 2
        self.diamond_radius = int(canvas_size * 0.35)  # 菱形半径
        
        logger.info(f"初始化Diamond Crop增强器")
        logger.info(f"  输入尺寸: {input_size}")
        logger.info(f"  画布尺寸: {canvas_size}×{canvas_size}")
        logger.info(f"  菱形半径: {self.diamond_radius}")
        
    def create_diamond_mask(self, angle=0):
        """
        创建菱形遮罩
        Args:
            angle: 旋转角度（度）
        Returns:
            mask: numpy数组，菱形区域为True
        """
        # 创建坐标网格
        y, x = np.ogrid[:self.canvas_size, :self.canvas_size]
        
        # 中心化坐标
        x = x - self.center
        y = y - self.center
        
        # 旋转坐标
        angle_rad = np.radians(angle)
        x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        
        # 菱形条件（曼哈顿距离）
        mask = (np.abs(x_rot) + np.abs(y_rot)) <= self.diamond_radius
        
        return mask
        
    def generate_multi_coverage_crops(self, 
                                    img_path, 
                                    num_angles=5,
                                    num_positions=4,
                                    to_grayscale=True,
                                    random_angles=False,
                                    random_seed=None):
        """
        生成多个菱形裁剪以覆盖整个图像
        
        Args:
            img_path: 输入图像路径
            num_angles: 每个位置的角度数量
            num_positions: 位置数量（4个角落）
            to_grayscale: 是否转换为灰度
            random_angles: 是否使用随机角度
            random_seed: 随机种子
            
        Returns:
            crops: 裁剪图像列表
            metadata: 元数据列表
        """
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        
        # 可选：转换为灰度
        if to_grayscale:
            img = img.convert('L').convert('RGB')
        
        # 调整到标准尺寸
        img = img.resize((self.input_w, self.input_h), Image.LANCZOS)
        img_array = np.array(img)
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
        crops = []
        metadata = []
        
        # 定义四个角落位置的偏移量
        positions = [
            (0, 0),           # 左上
            (0, 1),           # 右上
            (1, 0),           # 左下
            (1, 1)            # 右下
        ]
        
        # 对于每个位置
        for pos_idx, (offset_y, offset_x) in enumerate(positions):
            # 计算图像放置位置
            # 使图像的不同部分出现在菱形中心
            x_offset = int(offset_x * (self.canvas_size - self.input_w))
            y_offset = int(offset_y * (self.canvas_size - self.input_h))
            
            # 生成多个角度
            if random_angles:
                # 随机角度
                angles = np.random.uniform(0, 90, num_angles)
            else:
                # 均匀分布的角度
                angles = np.linspace(0, 90, num_angles, endpoint=False)
            
            for angle_idx, angle in enumerate(angles):
                # 创建空白画布
                canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
                
                # 放置图像
                canvas[y_offset:y_offset+self.input_h, 
                      x_offset:x_offset+self.input_w] = img_array
                
                # 创建菱形遮罩
                mask = self.create_diamond_mask(angle)
                
                # 应用遮罩（黑色背景）
                canvas[~mask] = 0
                
                # 保存裁剪结果
                crops.append(canvas)
                
                # 保存元数据
                meta = {
                    'position': f'pos_{pos_idx}',
                    'angle': float(angle),
                    'offset_x': x_offset,
                    'offset_y': y_offset,
                    'label': f'pos{pos_idx}_ang{angle_idx:02d}'
                }
                metadata.append(meta)
        
        return crops, metadata


class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, input_dir, output_dir, max_images=None, random_angles=True):
        """
        初始化批处理器
        Args:
            input_dir: 输入图像目录
            output_dir: 输出目录
            max_images: 最大处理图像数（None表示全部）
            random_angles: 是否使用随机角度
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_images = max_images
        self.random_angles = random_angles
        
        # 创建增强器
        self.augmentor = DiamondCropAugmentation(input_size=(720, 540), canvas_size=900)
        
        # 统计信息
        self.stats = {
            'total_images': 0,
            'processed': 0,
            'failed': 0,
            'total_crops': 0,
            'processing_time': 0,
            'failed_files': []
        }
        
    def get_image_files(self):
        """获取所有图像文件"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
        image_files = []
        
        logger.info(f"扫描目录: {self.input_dir}")
        for ext in extensions:
            files = list(self.input_dir.glob(ext))
            image_files.extend(files)
            if files:
                logger.info(f"  找到 {len(files)} 个 {ext} 文件")
        
        # 排序以保证处理顺序一致
        image_files.sort()
        
        # 限制数量
        if self.max_images:
            image_files = image_files[:self.max_images]
            logger.info(f"限制处理前 {self.max_images} 张图像")
        
        self.stats['total_images'] = len(image_files)
        return image_files
    
    def process_single_image(self, img_path, img_index):
        """
        处理单张图像
        Args:
            img_path: 图像路径
            img_index: 图像索引（用作随机种子）
        Returns:
            success: 是否成功
            crop_count: 生成的crop数量
        """
        try:
            # 创建输出目录（每张图像一个子目录）
            img_output_dir = self.output_dir / img_path.stem
            img_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 设置随机种子（每张图不同）
            seed = img_index if self.random_angles else None
            
            # 生成crops
            crops, metadata = self.augmentor.generate_multi_coverage_crops(
                img_path,
                to_grayscale=True,
                random_angles=self.random_angles,
                random_seed=seed
            )
            
            # 保存crops
            crop_count = 0
            for crop, meta in zip(crops, metadata):
                filename = f"{meta['label']}.png"
                filepath = img_output_dir / filename
                Image.fromarray(crop).save(filepath)
                crop_count += 1
            
            # 保存元数据
            meta_file = img_output_dir / 'metadata.json'
            with open(meta_file, 'w') as f:
                json.dump({
                    'source_image': str(img_path),
                    'num_crops': crop_count,
                    'crops': metadata
                }, f, indent=2)
            
            return True, crop_count
            
        except Exception as e:
            logger.error(f"处理 {img_path.name} 时出错: {e}")
            self.stats['failed_files'].append({
                'file': str(img_path),
                'error': str(e)
            })
            return False, 0
    
    def run(self):
        """运行批处理"""
        logger.info("\n" + "="*70)
        logger.info("批量Diamond Crop处理")
        logger.info("="*70)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取图像文件
        image_files = self.get_image_files()
        
        if not image_files:
            logger.error("未找到图像文件！")
            return
        
        logger.info(f"\n开始处理 {len(image_files)} 张图像...")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"角度模式: {'随机' if self.random_angles else '固定'}")
        logger.info("-"*70)
        
        # 开始计时
        start_time = time.time()
        
        # 处理进度条
        with tqdm(total=len(image_files), desc="总体进度", unit="图") as pbar:
            for idx, img_path in enumerate(image_files):
                # 更新描述
                pbar.set_description(f"处理 {img_path.name}")
                
                # 处理图像
                success, crop_count = self.process_single_image(img_path, idx)
                
                # 更新统计
                if success:
                    self.stats['processed'] += 1
                    self.stats['total_crops'] += crop_count
                else:
                    self.stats['failed'] += 1
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    '成功': self.stats['processed'],
                    '失败': self.stats['failed'],
                    '总crops': self.stats['total_crops']
                })
        
        # 计算总时间
        self.stats['processing_time'] = time.time() - start_time
        
        # 保存统计信息
        self.save_stats()
        
        # 打印最终统计
        self.print_summary()
    
    def save_stats(self):
        """保存处理统计信息"""
        stats_file = self.output_dir / 'processing_stats.json'
        
        # 添加时间戳
        self.stats['timestamp'] = datetime.now().isoformat()
        self.stats['input_dir'] = str(self.input_dir)
        self.stats['output_dir'] = str(self.output_dir)
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """打印处理摘要"""
        logger.info("\n" + "="*70)
        logger.info("处理完成！")
        logger.info("="*70)
        
        logger.info(f"\n📊 处理统计:")
        logger.info(f"  • 总图像数: {self.stats['total_images']}")
        logger.info(f"  • 成功处理: {self.stats['processed']}")
        logger.info(f"  • 处理失败: {self.stats['failed']}")
        logger.info(f"  • 生成crops: {self.stats['total_crops']}")
        
        if self.stats['processed'] > 0:
            avg_crops = self.stats['total_crops'] / self.stats['processed']
            logger.info(f"  • 每张图平均: {avg_crops:.1f} crops")
        
        # 时间统计
        total_time = self.stats['processing_time']
        if self.stats['processed'] > 0:
            avg_time = total_time / self.stats['processed']
        else:
            avg_time = 0
            
        logger.info(f"\n⏱️  时间统计:")
        logger.info(f"  • 总耗时: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)")
        logger.info(f"  • 平均每张: {avg_time:.2f} 秒")
        
        # 输出位置
        logger.info(f"\n📁 输出位置:")
        logger.info(f"  • Crops目录: {self.output_dir}")
        logger.info(f"  • 统计文件: {self.output_dir / 'processing_stats.json'}")
        
        # 失败文件
        if self.stats['failed_files']:
            logger.info(f"\n⚠️  失败文件 ({len(self.stats['failed_files'])} 个):")
            for item in self.stats['failed_files'][:5]:  # 只显示前5个
                logger.info(f"  • {Path(item['file']).name}: {item['error']}")
            if len(self.stats['failed_files']) > 5:
                logger.info(f"  • ... 还有 {len(self.stats['failed_files'])-5} 个")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量生成Diamond Crops用于SimCLR训练')
    
    # 输入输出参数
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入图像目录')
    parser.add_argument('--output', '-o', type=str, default='./crops_output',
                       help='输出目录 (默认: ./crops_output)')
    
    # 处理选项
    parser.add_argument('--max-images', '-m', type=int, default=None,
                       help='最大处理图像数 (默认: 全部)')
    parser.add_argument('--fixed-angles', action='store_true',
                       help='使用固定角度而非随机角度')
    
    # 测试模式
    parser.add_argument('--test', action='store_true',
                       help='测试模式：只处理前3张图像')
    
    args = parser.parse_args()
    
    # 验证输入目录
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"错误：输入目录不存在: {input_dir}")
        return
    
    # 测试模式
    if args.test:
        args.max_images = 3
        logger.info("🧪 测试模式：只处理前3张图像")
    
    # 创建处理器
    processor = BatchProcessor(
        input_dir=args.input,
        output_dir=args.output,
        max_images=args.max_images,
        random_angles=not args.fixed_angles
    )
    
    # 运行处理
    processor.run()
    
    logger.info("\n提示：crops生成完成后，使用以下命令训练ResNet+GeM模型：")
    logger.info(f"  python scripts/train_resnet_gem.py --crops_dir {args.output}")


if __name__ == "__main__":
    main()