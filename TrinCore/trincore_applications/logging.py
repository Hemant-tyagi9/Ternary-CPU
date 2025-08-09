import logging
import sys
from typing import Dict, Any
from datetime import datetime
from .config import TrinCoreConfig

class TernaryLogFormatter(logging.Formatter):
    """Custom formatter for ternary system logs"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[31;1m' # Bright red
    }
    RESET = '\033[0m'
    
    def format(self, record):
        level_color = self.COLORS.get(record.levelname, '')
        level_name = f"{level_color}{record.levelname}{self.RESET}"
        
        # Ternary state indicator
        state = TrinCoreConfig().get('system.mode', 'balanced')
        if state == 'balanced':
            state_icon = '⚖️'
        elif state == 'neuromorphic':
            state_icon = '🧠'
        else:
            state_icon = '🔧'
        
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        message = super().format(record)
        
        return f"{state_icon} [{timestamp}] {level_name:8} {message}"

class TrinCoreLogger:
    """Enhanced logging system for TrinCore"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrinCoreLogger, cls).__new__(cls)
            cls._instance._logger = logging.getLogger('TrinCore')
            cls._instance._setup_logger()
        return cls._instance
    
    def _setup_logger(self):
        config = TrinCoreConfig()
        self._logger.setLevel(config.get('system.log_level', 'INFO'))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(TernaryLogFormatter(
            fmt='%(name)s - %(message)s'
        ))
        self._logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('trincore.log')
        file_handler.setFormatter(logging.Formatter(
            fmt='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
        ))
        self._logger.addHandler(file_handler)
    
    def log_operation(self, operation: str, operands: tuple, result: Any):
        """Log ternary operations with details"""
        self._logger.debug(
            f"Operation: {operation}{operands} → {result} "
            f"[Ternary: {self._to_ternary_str(operands)} → {self._to_ternary_str((result,))}]"
        )
    
    def _to_ternary_str(self, values: tuple) -> str:
        """Convert values to ternary string representation"""
        mapping = {0: '0', 1: '1', 2: '2'}
        return "(" + ", ".join(mapping.get(v, '?') for v in values) + ")"
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying logger"""
        return getattr(self._logger, name)

# Global logger instance
logger = TrinCoreLogger()
