#!/usr/bin/env python3
"""
特征提取脚本
批量提取数据库和查询图像的特征
"""
import sys
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from src.feature.preprocessor import ImagePreprocessor
from src.feature.extractor import DinoV2Extractor
from src.feature.aggregator import VLADAggregator
from src.utils.logger import setup_logger


def extract_features(image_dir: str, 
                    output_file: str,
                    config: dict,
                    is_query: bool = False):
    """
    批量提取特征
    
    Args:
        image_dir: 图像目录
        output_file: 输出文件路径
        config: 配置字典
        is_query: 是否为查询图像
    """
    logger = setup_logger()
    device = config['model'].get('device', 'cuda')
    
    # 初始化组件
    preprocessor = ImagePreprocessor(
        target_size=config['preprocessing']['target_size'],
        patch_size=config['preprocessing']['patch_size']
    )
    
    extractor = DinoV2Extractor(
        model_name=config['model']['name'],
        layer=config['model']['layer'],
        device=device
    )
    
    aggregator = VLADAggregator(
        vocab_path=config['model']['vocab_path'],
        device=device
    )
    
    # 获取图像列表
    image_paths = sorted(Path(image_dir).glob("*.jpg"))
    logger.info(f"Found {len(image_paths)} images")
    
    # 提取特征
    all_descriptors = []
    all_ids = []
    
    for img_path in tqdm(image_paths, desc="Extracting features"):
        # 预处理
        img_tensor = preprocessor.preprocess(str(img_path), is_query=is_query)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # 特征提取
        with torch.no_grad():
            features = extractor.extract(img_tensor)
            descriptor = aggregator.aggregate(features[0])
            
            all_descriptors.append(descriptor.cpu())
            all_ids.append(img_path.stem)
        
        # 定期清理显存
        if len(all_descriptors) % 100 == 0 and device == 'cuda':
            torch.cuda.empty_cache()
    
    # 保存结果
    descriptors_tensor = torch.stack(all_descriptors)
    
    torch.save({
        'vlad_features': descriptors_tensor,
        'image_ids': all_ids,
        'image_paths': [str(p) for p in image_paths],
        'config': config,
        'timestamp': datetime.now().isoformat()
    }, output_file)
    
    logger.info(f"Saved descriptors to {output_file}")


def main():
    """主函数"""
    # 加载配置
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 提取数据库特征
    # extract_features(
    #     image_dir=config['data']['database_dir'],
    #     output_file=config['data']['output_dir'] + "/database_dinov2_vitb14_gray14_hardvlad.pt",
    #     config=config,
    #     is_query=False
    # )
    
    # 提取查询特征
    extract_features(
        image_dir=config['data']['query_dir'],
        output_file=config['data']['output_dir'] + "/query_dinov2_vitb14_optimized.pt",
        config=config,
        is_query=True
    )


if __name__ == "__main__":
    main()