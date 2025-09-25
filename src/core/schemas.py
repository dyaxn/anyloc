"""
接口和数据映射定义
"""
from typing import Dict, Any

# CSV列名到属性的映射
TITLES_MAP = {
    'Filename': 'filename',
    'Top_left_lat': 'top_left_lat',
    'Top_left_long': 'top_left_long',
    'Bottom_right_lat': 'bottom_right_lat',
    'Bottom_right_long': 'bottom_right_long',
    'zoom_level': 'zoom_level',
    'spatial_resolution': 'spatial_resolution'
}

TITLES_INFO = {
    'Filename': 'filename',
    'Top_left_lat': 'top_left_lat',
    'Top_left_long': 'top_left_long',
    'Bottom_right_lat': 'bottom_right_lat',
    'Bottom_right_long': 'bottom_right_long',
    'x': 'tile_x',
    'y': 'tile_y',
    'zoom_level': 'zoom_level',
    'spatial_resolution': 'spatial_resolution',
    'confidence': 'confidence'
}