"""
日志配置模块
"""
import logging
import sys
from pathlib import Path
from typing import Optional 


def setup_logger(name: str = "aerial_map_retrieval", 
                log_file: Optional[str] = None,
                level: int = logging.INFO) -> logging.Logger:
    """
    配置日志系统
    
    Args:
        name: 日志名称
        log_file: 日志文件路径
        level: 日志级别
    
    Returns:
        配置好的logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger