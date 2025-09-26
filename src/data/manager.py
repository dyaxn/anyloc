"""
数据管理模块
管理元数据的加载和查询
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging

from src.core.models import TileMetadata
from src.core.schemas import TITLES_MAP

logger = logging.getLogger(__name__)


class MetadataManager:
    """元数据管理器"""
    
    def __init__(self):
        self.metadata_dict: Dict[str, TileMetadata] = {}
    
    def load_from_csv(self, csv_path: str):
        """
        从CSV文件加载元数据
        
        Args:
            csv_path: CSV文件路径
        """
        logger.info(f"Loading metadata from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        for idx, row in df.iterrows():
            # 处理文件名
            filename_value = row.get('Filename', row.get('filename', ''))
            if isinstance(filename_value, (int, float)):
                file_id = str(int(filename_value))
            else:
                file_id = Path(str(filename_value)).stem
            
            # 创建元数据对象
            metadata = TileMetadata(
                filename=f"{file_id}.jpg",
                top_left_lat=float(row.get('Top_left_lat', 0)),
                top_left_long=float(row.get('Top_left_long', 0)),
                bottom_right_lat=float(row.get('Bottom_right_lat', 0)),
                bottom_right_long=float(row.get('Bottom_right_long', 0)),
                zoom_level=int(row.get('zoom_level', 18)),
                spatial_resolution=float(row.get('spatial_resolution', 0)),
                confidence=float(row.get('confidence', -1))
            )
            
            self.metadata_dict[file_id] = metadata
        
        logger.info(f"Loaded {len(self.metadata_dict)} metadata entries")
    
    def get_metadata(self, tile_id: str) -> Optional[TileMetadata]:
        """
        获取指定瓦片的元数据
        
        Args:
            tile_id: 瓦片ID
        
        Returns:
            元数据对象，如果不存在返回None
        """
        return self.metadata_dict.get(tile_id)
    
    def add_metadata(self, tile_id: str, metadata: TileMetadata):
        """添加元数据"""
        self.metadata_dict[tile_id] = metadata
    
    @property
    def size(self) -> int:
        """获取元数据数量"""
        return len(self.metadata_dict)