"""Configuration loader for TennisPreview"""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class Config:
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self.load()
    
    def load(self, config_path: Optional[str] = None):
        """Load configuration from YAML file and environment variables"""
        load_dotenv()
        
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Override with environment variables
        self._apply_env_overrides()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides for API keys"""
        env_mapping = {
            'THEODDSAPI_KEY': ('apis', 'theoddsapi', 'api_key'),
            'API_FOOTBALL_KEY': ('apis', 'api_football', 'api_key'),
            'BETFAIR_APP_KEY': ('apis', 'betfair', 'app_key'),
            'BETFAIR_SESSION_TOKEN': ('apis', 'betfair', 'session_token'),
        }
        
        for env_var, config_path in env_mapping.items():
            value = os.getenv(env_var)
            if not value:
                continue
            # Ignore placeholder values from .env.example
            if 'your_' in value.lower() and 'here' in value.lower():
                continue
            if value.strip().lower() in ('your_key_here', 'your_app_key_here',
                                         'your_session_token_here'):
                continue
            self.set(*config_path, value=value)
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """Get nested config value using dot notation keys"""
        current = self._config
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return default
            if current is None:
                return default
        return current
    
    def set(self, *keys: str, value: Any):
        """Set nested config value"""
        current = self._config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    @property
    def all(self) -> Dict[str, Any]:
        return self._config.copy()


config = Config()