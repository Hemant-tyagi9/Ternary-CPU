import json
import os
from typing import Dict, Any

class TrinCoreConfig:
    """Enhanced configuration system for TrinCore"""
    
    _instance = None
    DEFAULT_CONFIG = {
        "system": {
            "mode": "balanced",  # balanced, traditional, neural, neuromorphic
            "log_level": "INFO",
            "ternary_representation": "balanced",
            "max_parallel_ops": 4
        },
        "neuromorphic": {
            "event_threads": 4,
            "memory_computing": True,
            "adaptation_rate": 0.1
        },
        "nn": {
            "batch_size": 32,
            "vectorize_ops": True,
            "model_cache": "models/"
        }
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrinCoreConfig, cls).__new__(cls)
            cls._instance._config = cls.DEFAULT_CONFIG.copy()
            cls._instance._config_path = "trincore_config.json"
            cls._instance.load()
        return cls._instance
    
    def load(self):
        """Load configuration from file if exists"""
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, 'r') as f:
                    self._config = json.load(f)
                # Merge with defaults for any missing keys
                for section, values in self.DEFAULT_CONFIG.items():
                    if section not in self._config:
                        self._config[section] = values
                    else:
                        for k, v in values.items():
                            if k not in self._config[section]:
                                self._config[section][k] = v
            except Exception as e:
                print(f"⚠️ Config load failed, using defaults: {e}")
    
    def save(self):
        """Save current configuration to file"""
        try:
            with open(self._config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            print(f"⚠️ Config save failed: {e}")
    
    def get(self, key_path: str, default=None) -> Any:
        """Get config value using dot notation (e.g., 'system.mode')"""
        keys = key_path.split('.')
        current = self._config
        try:
            for key in keys:
                current = current[key]
            return current
        except KeyError:
            return default
    
    def set(self, key_path: str, value):
        """Set config value using dot notation"""
        keys = key_path.split('.')
        current = self._config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
        self.save()
    
    def __str__(self):
        return json.dumps(self._config, indent=2)
