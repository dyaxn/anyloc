"""
数据模型定义
统一管理所有数据结构
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class TileMetadata:
    """地图瓦片元数据"""
    filename: str
    top_left_lat: float
    top_left_long: float
    bottom_right_lat: float
    bottom_right_long: float
    zoom_level: int = 18
    spatial_resolution: float = 0.0
    confidence: float = -1.0
    tile_x: int = 0
    tile_y: int = 0
    
    @property
    def center_lat(self) -> float:
        """计算中心纬度"""
        return (self.top_left_lat + self.bottom_right_lat) / 2
    
    @property
    def center_long(self) -> float:
        """计算中心经度"""
        return (self.top_left_long + self.bottom_right_long) / 2
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'filename': self.filename,
            'top_left_lat': self.top_left_lat,
            'top_left_long': self.top_left_long,
            'bottom_right_lat': self.bottom_right_lat,
            'bottom_right_long': self.bottom_right_long,
            'center_lat': self.center_lat,
            'center_long': self.center_long,
            'zoom_level': self.zoom_level,
            'spatial_resolution': self.spatial_resolution,
            'confidence': self.confidence
        }


@dataclass
class RetrievalResult:
    """检索结果"""
    rank: int
    tile_metadata: TileMetadata
    similarity_score: float
    descriptor_idx: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            'rank': self.rank,
            'similarity_score': self.similarity_score,
            'filename': self.tile_metadata.filename
        }
        result.update(self.tile_metadata.to_dict())
        return result


@dataclass
class DescriptorData:
    """描述符数据"""
    features: np.ndarray
    image_ids: list
    image_paths: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)