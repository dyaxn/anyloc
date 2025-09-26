#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Retrieval Engine
支持不同维度的描述符和多种检索策略
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import logging
import json
from datetime import datetime

from .indexer import FAISSIndexer
from ..data.manager import MetadataManager
from ..feature.extractor import FeatureExtractor
from ..feature.aggregator import FeatureAggregator
from ..feature.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    增强版检索引擎
    支持多种模型和聚合方法的组合
    """
    
    def __init__(self, config: dict):
        """
        初始化检索引擎
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = config.get('system', {}).get('device', 'cuda')
        
        # 组件初始化标志
        self.extractor = None
        self.aggregator = None
        self.preprocessor = None
        self.indexer = None
        self.metadata_manager = None
        
        # 数据库信息
        self.db_descriptors = None
        self.db_image_ids = None
        self.db_metadata = None
        
        # 性能统计
        self.stats = {
            'queries_processed': 0,
            'total_time': 0,
            'avg_time': 0
        }
        
        logger.info("检索引擎初始化完成")
        
    def setup_components(self, model_type: str = None, agg_type: str = None):
        """
        设置检索组件
        
        Args:
            model_type: 模型类型（覆盖配置）
            agg_type: 聚合类型（覆盖配置）
        """
        # 获取模型和聚合器设置
        if model_type:
            self.config['model']['extractor']['type'] = model_type
        if agg_type:
            self.config['model']['aggregator']['type'] = agg_type
            
        model_config = self.config['model']['extractor']
        agg_config = self.config['model']['aggregator']
        
        # 初始化特征提取器
        logger.info(f"初始化特征提取器: {model_config['type']}")
        self.extractor = FeatureExtractor(
            model_type=model_config['type'],
            layer=model_config.get('layer', 31),
            facet=model_config.get('facet', 'value'),
            device=self.device,
            checkpoint_path=model_config.get('checkpoint_path')
        )
        
        # 初始化聚合器
        logger.info(f"初始化聚合器: {agg_config['type']}")
        if agg_config['type'] == 'vlad':
            # 获取词汇表路径
            vocab_path = self._get_vocab_path(model_config['type'])
            self.aggregator = FeatureAggregator(
                agg_type='vlad',
                vocab_path=vocab_path,
                device=self.device
            )
        else:
            self.aggregator = FeatureAggregator(
                agg_type='gem',
                gem_p=agg_config['gem'].get('p', 3.0),
                device=self.device
            )
        
        # 初始化预处理器
        self.preprocessor = ImagePreprocessor(
            target_size=None,
            mode='query',
            patch_size=self.extractor.get_patch_size()
        )
        
        # 初始化索引器
        use_gpu = self.config['retrieval']['index'].get('use_gpu', True)
        self.indexer = FAISSIndexer(use_gpu=(use_gpu and self.device == 'cuda'))
        
    def _get_vocab_path(self, model_type: str) -> str:
        """
        获取模型对应的词汇表路径
        
        Args:
            model_type: 模型类型
            
        Returns:
            词汇表路径
        """
        vlad_config = self.config['model']['aggregator']['vlad']
        
        # 检查模型特定的词汇表
        vocab_paths = vlad_config.get('vocab_paths', {})
        if model_type in vocab_paths:
            return vocab_paths[model_type]
        
        # 使用默认词汇表
        return vlad_config.get('vocab_path')
    
    def load_database(self, 
                     descriptor_file: Union[str, Path] = None,
                     metadata_file: Union[str, Path] = None,
                     auto_detect: bool = True) -> None:
        """
        加载数据库描述符和元数据
        
        Args:
            descriptor_file: 描述符文件路径
            metadata_file: 元数据CSV文件路径
            auto_detect: 是否自动检测描述符文件
        """
        # 自动检测描述符文件
        if auto_detect and descriptor_file is None:
            descriptor_file = self._auto_detect_descriptor_file()
        
        if descriptor_file is None:
            raise ValueError("未指定描述符文件且自动检测失败")
        
        descriptor_file = Path(descriptor_file)
        if not descriptor_file.exists():
            raise FileNotFoundError(f"描述符文件不存在: {descriptor_file}")
        
        # 加载描述符
        logger.info(f"加载数据库描述符: {descriptor_file}")
        data = torch.load(descriptor_file, map_location='cpu')
        
        # 处理不同格式
        if isinstance(data, dict):
            # 新格式
            if 'descriptors' in data:
                self.db_descriptors = data['descriptors']
            elif 'features' in data:
                self.db_descriptors = data['features']
            elif 'vlad_features' in data:
                self.db_descriptors = data['vlad_features']
            else:
                raise KeyError("无法找到描述符数据")
            
            self.db_image_ids = data.get('image_ids', [])
            
            # 记录元数据
            if 'metadata' in data:
                self.db_metadata = data['metadata']
                logger.info(f"数据库元数据: {self.db_metadata}")
        else:
            # 旧格式：直接tensor
            self.db_descriptors = data
            self.db_image_ids = []
        
        # 转换为numpy
        if isinstance(self.db_descriptors, torch.Tensor):
            self.db_descriptors = self.db_descriptors.numpy()
        
        logger.info(f"  描述符形状: {self.db_descriptors.shape}")
        logger.info(f"  图像数量: {len(self.db_image_ids) if self.db_image_ids else self.db_descriptors.shape[0]}")
        
        # 构建索引
        logger.info("构建FAISS索引...")
        self.indexer.build_index(self.db_descriptors)
        
        # 加载元数据（如果提供）
        if metadata_file:
            metadata_file = Path(metadata_file)
            if metadata_file.exists():
                logger.info(f"加载元数据: {metadata_file}")
                self.metadata_manager = MetadataManager(metadata_file)
            else:
                logger.warning(f"元数据文件不存在: {metadata_file}")
    
    def _auto_detect_descriptor_file(self) -> Optional[Path]:
        """
        自动检测描述符文件
        
        Returns:
            描述符文件路径或None
        """
        desc_dir = Path(self.config['data']['descriptor_dir'])
        
        if not desc_dir.exists():
            return None
        
        # 根据当前模型和聚合器生成文件名模式
        model_type = self.config['model']['extractor']['type']
        agg_type = self.config['model']['aggregator']['type']
        
        model_suffix = model_type.replace('/', '_')
        pattern = f"database_*{model_suffix}*{agg_type}*.pt"
        
        # 搜索匹配的文件
        matches = list(desc_dir.glob(pattern))
        
        if matches:
            # 返回最新的文件
            latest = max(matches, key=lambda p: p.stat().st_mtime)
            logger.info(f"自动检测到描述符文件: {latest}")
            return latest
        
        # 尝试更宽松的匹配
        pattern = f"database_*.pt"
        matches = list(desc_dir.glob(pattern))
        
        if matches:
            logger.warning(f"使用通用描述符文件: {matches[0]}")
            return matches[0]
        
        return None
    
    def query(self, 
             query_image: Union[str, Path, np.ndarray],
             top_k: int = 10) -> List[Dict]:
        """
        执行单张图像检索
        
        Args:
            query_image: 查询图像（路径或数组）
            top_k: 返回前K个结果
            
        Returns:
            检索结果列表
        """
        if self.db_descriptors is None:
            raise RuntimeError("数据库未加载，请先调用load_database()")
        
        # 提取查询图像特征
        query_descriptor = self._extract_query_descriptor(query_image)
        
        # 执行检索
        similarities, indices = self.indexer.search(
            query_descriptor.reshape(1, -1), 
            top_k
        )
        
        # 构建结果
        results = []
        for rank, (sim, idx) in enumerate(zip(similarities[0], indices[0]), 1):
            result = {
                'rank': rank,
                'score': float(sim),
                'index': int(idx)
            }
            
            # 添加图像ID
            if self.db_image_ids and idx < len(self.db_image_ids):
                result['image_id'] = self.db_image_ids[idx]
            else:
                result['image_id'] = str(idx)
            
            # 添加元数据
            if self.metadata_manager:
                metadata = self.metadata_manager.get_metadata(result['image_id'])
                if metadata:
                    result.update(metadata)
            
            results.append(result)
        
        # 更新统计
        self.stats['queries_processed'] += 1
        
        return results
    
    def batch_query(self, 
                   query_images: List[Union[str, Path]],
                   top_k: int = 10) -> List[List[Dict]]:
        """
        批量查询
        
        Args:
            query_images: 查询图像列表
            top_k: 每个查询返回前K个结果
            
        Returns:
            所有查询的结果列表
        """
        all_results = []
        
        for query_image in query_images:
            results = self.query(query_image, top_k)
            all_results.append(results)
        
        return all_results
    
    def _extract_query_descriptor(self, query_image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        提取查询图像的描述符
        
        Args:
            query_image: 查询图像
            
        Returns:
            描述符向量
        """
        # 处理不同输入类型
        if isinstance(query_image, (str, Path)):
            # 从文件路径加载
            img_tensor = self.preprocessor.process(query_image)
        elif isinstance(query_image, np.ndarray):
            # 从numpy数组
            from PIL import Image
            img = Image.fromarray(query_image)
            img_tensor = self.preprocessor.process_pil(img)
        else:
            raise TypeError(f"不支持的查询图像类型: {type(query_image)}")
        
        # 添加batch维度
        img_batch = img_tensor.unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.extractor.extract(img_batch)
            
            # 聚合特征
            if self.aggregator.agg_type == "vlad":
                # VLAD需要单独处理
                descriptor = self.aggregator.aggregate(features[0])
            else:
                # GeM可以处理batch
                descriptor = self.aggregator.aggregate(features)
                if descriptor.dim() > 1:
                    descriptor = descriptor[0]
            
            descriptor = descriptor.cpu().numpy()
        
        return descriptor
    
    def evaluate(self, 
                ground_truth: Dict[str, List[str]],
                top_k_list: List[int] = [1, 5, 10]) -> Dict:
        """
        评估检索性能
        
        Args:
            ground_truth: 真值字典 {query_id: [relevant_ids]}
            top_k_list: 要评估的K值列表
            
        Returns:
            评估指标字典
        """
        metrics = {f'recall@{k}': [] for k in top_k_list}
        
        for query_id, relevant_ids in ground_truth.items():
            # 执行检索
            results = self.query(query_id, max(top_k_list))
            
            # 获取检索到的ID
            retrieved_ids = [r['image_id'] for r in results]
            
            # 计算各个K值的召回率
            for k in top_k_list:
                top_k_ids = retrieved_ids[:k]
                # 检查是否有相关文档被检索到
                hit = any(rid in relevant_ids for rid in top_k_ids)
                metrics[f'recall@{k}'].append(1.0 if hit else 0.0)
        
        # 计算平均值
        avg_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                avg_metrics[metric_name] = np.mean(values)
            else:
                avg_metrics[metric_name] = 0.0
        
        return avg_metrics
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.stats.copy()
        
        # 添加数据库信息
        if self.db_descriptors is not None:
            stats['database_size'] = self.db_descriptors.shape[0]
            stats['descriptor_dim'] = self.db_descriptors.shape[1]
        
        # 添加配置信息
        stats['model'] = self.config['model']['extractor']['type']
        stats['aggregator'] = self.config['model']['aggregator']['type']
        
        return stats
    
    def save_index(self, save_path: Union[str, Path]) -> None:
        """
        保存索引到文件
        
        Args:
            save_path: 保存路径
        """
        if self.indexer is None or self.indexer.index is None:
            raise RuntimeError("索引未构建")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        import faiss
        faiss.write_index(self.indexer.index, str(save_path))
        
        # 保存元数据
        metadata_path = save_path.parent / f"{save_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'model': self.config['model']['extractor']['type'],
                'aggregator': self.config['model']['aggregator']['type'],
                'database_size': self.db_descriptors.shape[0] if self.db_descriptors is not None else 0,
                'descriptor_dim': self.db_descriptors.shape[1] if self.db_descriptors is not None else 0,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"索引已保存到: {save_path}")
    
    def load_index(self, index_path: Union[str, Path]) -> None:
        """
        从文件加载索引
        
        Args:
            index_path: 索引文件路径
        """
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
        
        # 加载FAISS索引
        import faiss
        self.indexer.index = faiss.read_index(str(index_path))
        
        logger.info(f"索引已从 {index_path} 加载")


class MultiModelRetrieval:
    """
    多模型融合检索
    支持多个模型的结果融合
    """
    
    def __init__(self, config: dict):
        """
        初始化多模型检索
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.engines = {}
        
        # 融合方法
        self.fusion_method = config.get('retrieval', {}).get('ensemble', {}).get('fusion_method', 'weighted')
        self.weights = config.get('retrieval', {}).get('ensemble', {}).get('weights', [])
        
    def add_engine(self, name: str, engine: RetrievalEngine, weight: float = 1.0):
        """
        添加检索引擎
        
        Args:
            name: 引擎名称
            engine: 检索引擎实例
            weight: 融合权重
        """
        self.engines[name] = {
            'engine': engine,
            'weight': weight
        }
        logger.info(f"添加检索引擎: {name} (权重: {weight})")
    
    def query(self, query_image: Union[str, Path], top_k: int = 10) -> List[Dict]:
        """
        多模型融合查询
        
        Args:
            query_image: 查询图像
            top_k: 返回前K个结果
            
        Returns:
            融合后的检索结果
        """
        if not self.engines:
            raise RuntimeError("未添加任何检索引擎")
        
        # 收集所有引擎的结果
        all_results = {}
        for name, engine_info in self.engines.items():
            engine = engine_info['engine']
            weight = engine_info['weight']
            
            # 获取更多结果用于融合
            results = engine.query(query_image, top_k * 3)
            all_results[name] = {
                'results': results,
                'weight': weight
            }
        
        # 融合结果
        if self.fusion_method == 'weighted':
            return self._weighted_fusion(all_results, top_k)
        elif self.fusion_method == 'rank':
            return self._rank_fusion(all_results, top_k)
        else:
            raise ValueError(f"不支持的融合方法: {self.fusion_method}")
    
    def _weighted_fusion(self, all_results: Dict, top_k: int) -> List[Dict]:
        """
        加权分数融合
        
        Args:
            all_results: 所有引擎的结果
            top_k: 返回前K个
            
        Returns:
            融合后的结果
        """
        # 收集所有唯一的图像ID及其加权分数
        fusion_scores = {}
        
        for engine_name, engine_data in all_results.items():
            results = engine_data['results']
            weight = engine_data['weight']
            
            for result in results:
                img_id = result['image_id']
                score = result['score'] * weight
                
                if img_id not in fusion_scores:
                    fusion_scores[img_id] = {
                        'total_score': 0,
                        'contributions': {},
                        'metadata': result  # 保存元数据
                    }
                
                fusion_scores[img_id]['total_score'] += score
                fusion_scores[img_id]['contributions'][engine_name] = result['score']
        
        # 排序并返回top-k
        sorted_items = sorted(
            fusion_scores.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )[:top_k]
        
        # 构建最终结果
        final_results = []
        for rank, (img_id, data) in enumerate(sorted_items, 1):
            result = data['metadata'].copy()
            result['rank'] = rank
            result['score'] = data['total_score']
            result['fusion_contributions'] = data['contributions']
            final_results.append(result)
        
        return final_results
    
    def _rank_fusion(self, all_results: Dict, top_k: int) -> List[Dict]:
        """
        排名融合（Reciprocal Rank Fusion）
        
        Args:
            all_results: 所有引擎的结果
            top_k: 返回前K个
            
        Returns:
            融合后的结果
        """
        # RRF常数
        k_const = 60
        
        # 收集所有图像的RRF分数
        rrf_scores = {}
        
        for engine_name, engine_data in all_results.items():
            results = engine_data['results']
            
            for result in results:
                img_id = result['image_id']
                rank = result['rank']
                
                # 计算RRF分数
                rrf_score = 1.0 / (k_const + rank)
                
                if img_id not in rrf_scores:
                    rrf_scores[img_id] = {
                        'total_rrf': 0,
                        'metadata': result
                    }
                
                rrf_scores[img_id]['total_rrf'] += rrf_score
        
        # 排序并返回top-k
        sorted_items = sorted(
            rrf_scores.items(),
            key=lambda x: x[1]['total_rrf'],
            reverse=True
        )[:top_k]
        
        # 构建最终结果
        final_results = []
        for rank, (img_id, data) in enumerate(sorted_items, 1):
            result = data['metadata'].copy()
            result['rank'] = rank
            result['score'] = data['total_rrf']
            final_results.append(result)
        
        return final_results