"""Logging setup with progress bars for TennisPreview"""
import sys
import logging
from pathlib import Path
from typing import Optional
from loguru import logger
from tqdm import tqdm
import threading
import time


class ProgressTracker:
    """Thread-safe progress tracker with a single reusable bar"""
    
    def __init__(self):
        self._enabled = True
        self._lock = threading.Lock()
        self._main_bar: Optional[tqdm] = None
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if not enabled:
            self.close_all()
    
    def start(self, total: int, desc: str = "Processing") -> tqdm:
        """Start the main progress bar (closes previous one if any)"""
        with self._lock:
            if self._main_bar is not None:
                try:
                    self._main_bar.close()
                except Exception:
                    pass
            if self._enabled:
                self._main_bar = tqdm(
                    total=total,
                    desc=desc,
                    ncols=100,
                    leave=True,
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                )
            else:
                self._main_bar = None
            return self._main_bar
    
    def update(self, n: int = 1):
        """Update main progress bar"""
        with self._lock:
            if self._main_bar is not None:
                self._main_bar.update(n)
    
    def set_desc(self, desc: str):
        """Set main progress bar description"""
        with self._lock:
            if self._main_bar is not None:
                self._main_bar.set_description(desc)
    
    def write(self, msg: str):
        """Write a log line above the progress bar without breaking it"""
        with self._lock:
            if self._main_bar is not None:
                self._main_bar.clear()
            print(msg)
            if self._main_bar is not None:
                self._main_bar.display()
    
    def close(self):
        """Close the main progress bar"""
        with self._lock:
            if self._main_bar is not None:
                self._main_bar.close()
                self._main_bar = None
    
    def close_all(self):
        self.close()
    
    # Backward-compatible aliases used across modules
    def start_main(self, total: int, desc: str = "Processing") -> tqdm:
        return self.start(total, desc)
    
    def update_main(self, n: int = 1):
        self.update(n)
    
    def set_main_desc(self, desc: str):
        self.set_desc(desc)
    
    def start_sub(self, key: str, total: int, desc: str = "") -> tqdm:
        return self.start(total, desc or key)
    
    def update_sub(self, key: str, n: int = 1):
        self.update(n)
    
    def close_sub(self, key: str):
        self.close()


# Global progress tracker
progress = ProgressTracker()


def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """Setup loguru logging with file rotation and console output"""
    
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True
    )
    
    # File handler with rotation
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_path / "tennispreview_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",
        retention="30 days",
        compression="zip"
    )
    
    # Error file
    logger.add(
        log_path / "errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",
        retention="90 days"
    )
    
    return logger


def get_logger(name: str):
    """Get logger instance for module"""
    return logger.bind(module=name)