#!/usr/bin/env python3
"""
检索系统执行脚本
"""
import sys
import json
import yaml
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval.engine import RetrievalEngine
from src.utils.logger import setup_logger


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger()
    
    # 加载配置
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = load_config(str(config_path))
    
    # 初始化检索引擎
    engine = RetrievalEngine(config)
    
    # 加载数据库
    engine.load_database(
        descriptor_file=config['data']['output_dir'] + "/database_descriptors_optimized.pt",
        metadata_file=config['data']['metadata_file']
    )
    
    # 预热模型
    if config['model'].get('warmup_iterations', 0) > 0:
        engine.warmup(config['model']['warmup_iterations'])
    
    # 执行查询
    query_image = "/root/autodl-tmp/data/target_drone_img/305383811000.jpg"
    
    # 带计时的查询
    results, timing = engine.query_with_timing(query_image, top_k=10)
    
    # 打印计时信息
    print("\n" + "="*60)
    print("Performance Analysis:")
    print("="*60)
    for step, time_ms in timing.items():
        print(f"{step:20s}: {time_ms*1000:7.2f} ms")
    
    # 保存结果
    output_file = Path(config['data']['output_dir']) / "retrieval_results.json"
    results_dict = [r.to_dict() for r in results]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Top match: {results[0].tile_metadata.filename} (score: {results[0].similarity_score:.4f})")


if __name__ == "__main__":
    main()