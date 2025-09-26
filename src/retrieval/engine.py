"""
检索引擎
整合特征提取、索引和元数据管理
"""
import time
import torch
import numpy as np  
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

from src.core.models import TileMetadata, RetrievalResult
from src.feature.preprocessor import ImagePreprocessor
from src.feature.extractor import DinoV2Extractor
from src.feature.aggregator import VLADAggregator
from src.retrieval.indexer import FAISSIndexer
from src.data.manager import MetadataManager

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """地图瓦片检索引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化检索引擎
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = config['model'].get('device', 'cuda')
        
        # 初始化各组件
        self.preprocessor = ImagePreprocessor(
            target_size=config['preprocessing']['target_size'],
            patch_size=config['preprocessing']['patch_size']
        )
        
        self.extractor = DinoV2Extractor(
            model_name=config['model']['name'],
            layer=config['model']['layer'],
            device=self.device
        )
        
        self.aggregator = VLADAggregator(
            vocab_path=config['model']['vocab_path'],
            device=self.device
        )
        
        self.indexer = None
        self.metadata_manager = MetadataManager()
        self.db_ids = []
        
    def load_database(self, descriptor_file: str, metadata_file: Optional[str] = None):
        """
        加载数据库
        
        Args:
            descriptor_file: 描述符文件路径
            metadata_file: 元数据文件路径
        """
        logger.info(f"Loading database from {descriptor_file}")
        
        # 加载描述符
        data = torch.load(descriptor_file, map_location='cpu')
        db_descriptors = data['vlad_features']
        self.db_ids = data['image_ids']
        
        # 构建索引
        self.indexer = FAISSIndexer(
            dimension=db_descriptors.shape[1],
            use_gpu=self.config['retrieval']['use_gpu']
        )
        self.indexer.build_index(db_descriptors)
        
        # 加载元数据
        if metadata_file:
            self.metadata_manager.load_from_csv(metadata_file)
        
        logger.info(f"Database loaded: {len(self.db_ids)} images")
    
    def query(self, image_path: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        执行检索查询
        
        Args:
            image_path: 查询图像路径
            top_k: 返回top-k个结果
        
        Returns:
            检索结果列表
        """
        logger.info(f"Processing query: {image_path}")
        
        # 提取查询描述符
        query_descriptor = self._extract_descriptor(image_path, is_query=True)
        
        # 检索
        distances, indices = self.indexer.search(
            query_descriptor.unsqueeze(0), top_k
        )
        
        # 构建结果
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
            tile_id = self.db_ids[idx]
            
            # 获取元数据
            metadata = self.metadata_manager.get_metadata(tile_id)
            if metadata is None:
                # 创建默认元数据
                metadata = TileMetadata(
                    filename=f"{tile_id}.jpg",
                    top_left_lat=0.0,
                    top_left_long=0.0,
                    bottom_right_lat=0.0,
                    bottom_right_long=0.0
                )
            
            result = RetrievalResult(
                rank=rank + 1,
                tile_metadata=metadata,
                similarity_score=float(score),
                descriptor_idx=int(idx)
            )
            results.append(result)
        
        logger.info(f"Retrieved {len(results)} results")
        return results
    
    def query_with_timing(self, image_path: str, top_k: int = 10) -> Tuple[List[RetrievalResult], Dict[str, float]]:
        """
        带计时的检索查询
        
        Returns:
            (results, timing_info): 检索结果和计时信息
        """
        timing = {}
        total_start = time.time()
        
        # 预处理计时
        preprocess_start = time.time()
        img_tensor = self.preprocessor.preprocess(image_path, is_query=True)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        timing['preprocess'] = time.time() - preprocess_start
        
        # 特征提取计时
        extract_start = time.time()
        features = self.extractor.extract(img_tensor)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        timing['feature_extract'] = time.time() - extract_start
        
        # VLAD聚合计时
        vlad_start = time.time()
        descriptor = self.aggregator.aggregate(features[0])
        if self.device == 'cuda':
            torch.cuda.synchronize()
        timing['vlad_aggregate'] = time.time() - vlad_start
        
        # 检索计时
        search_start = time.time()
        distances, indices = self.indexer.search(descriptor.unsqueeze(0), top_k)
        timing['search'] = time.time() - search_start
        
        # 构建结果
        results = self._build_results(distances[0], indices[0])
        
        timing['total'] = time.time() - total_start
        
        return results, timing
    
    def _extract_descriptor(self, image_path: str, is_query: bool = False) -> torch.Tensor:
        """提取图像描述符"""
        # 预处理
        img_tensor = self.preprocessor.preprocess(image_path, is_query=is_query)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # 特征提取
        features = self.extractor.extract(img_tensor)
        
        # VLAD聚合
        descriptor = self.aggregator.aggregate(features[0])
        
        return descriptor
    
    def _build_results(self, distances: np.ndarray, indices: np.ndarray) -> List[RetrievalResult]:
        """构建检索结果"""
        results = []
        for rank, (idx, score) in enumerate(zip(indices, distances)):
            tile_id = self.db_ids[idx]
            metadata = self.metadata_manager.get_metadata(tile_id)
            
            if metadata is None:
                metadata = TileMetadata(
                    filename=f"{tile_id}.jpg",
                    top_left_lat=0.0,
                    top_left_long=0.0,
                    bottom_right_lat=0.0,
                    bottom_right_long=0.0
                )
            
            result = RetrievalResult(
                rank=rank + 1,
                tile_metadata=metadata,
                similarity_score=float(score),
                descriptor_idx=int(idx)
            )
            results.append(result)
        
        return results
    
    def warmup(self, iterations: int = 3):
        """模型预热"""
        logger.info("Warming up model")
        self.extractor.init_model()
        
        dummy_img = torch.randn(1, 3, 896, 896).to(self.device)
        
        for i in range(iterations):
            start = time.time()
            with torch.no_grad():
                features = self.extractor.extract(dummy_img)
                descriptor = self.aggregator.aggregate(features[0])
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            logger.info(f"Warmup iteration {i+1}: {elapsed*1000:.2f}ms")