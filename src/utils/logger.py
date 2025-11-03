"""
日志工具模块
提供统一的日志管理
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(name: str = 'xgboost_stock',
                log_dir: str = 'logs',
                level: int = logging.INFO,
                console: bool = True,
                file: bool = True) -> logging.Logger:
    """
    设置日志器
    
    Args:
        name: 日志器名称
        log_dir: 日志目录
        level: 日志级别
        console: 是否输出到控制台
        file: 是否输出到文件
        
    Returns:
        Logger对象
    """
    # 创建日志目录
    if file:
        os.makedirs(log_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 添加文件handler
    if file:
        log_file = os.path.join(
            log_dir,
            f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"日志文件: {log_file}")
    
    return logger


def get_logger(name: str = 'xgboost_stock') -> logging.Logger:
    """
    获取已存在的logger
    
    Args:
        name: 日志器名称
        
    Returns:
        Logger对象
    """
    return logging.getLogger(name)


class LoggerContext:
    """日志上下文管理器"""
    
    def __init__(self, logger: logging.Logger, message: str, level: int = logging.INFO):
        """
        初始化
        
        Args:
            logger: Logger对象
            message: 开始消息
            level: 日志级别
        """
        self.logger = logger
        self.message = message
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.level, f"开始: {self.message}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.log(self.level, f"完成: {self.message} (耗时: {elapsed:.2f}秒)")
        else:
            self.logger.error(f"失败: {self.message} (耗时: {elapsed:.2f}秒)", exc_info=True)
        
        return False  # 不抑制异常


# 测试代码
if __name__ == "__main__":
    # 测试基本日志
    logger = setup_logger('test_logger', log_dir='logs')
    
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    
    # 测试上下文管理器
    with LoggerContext(logger, "测试任务"):
        import time
        time.sleep(1)
        logger.info("任务执行中...")
    
    print("日志测试完成")
