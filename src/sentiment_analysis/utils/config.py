"""
Configuration management for sentiment analysis application.

This module provides utilities for loading and managing
application configuration from files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """
    Configuration manager for the application.

    This class handles loading configuration from YAML files
    and environment variables.

    Example:
        >>> config = Config.load("configs/model_config.yaml")
        >>> print(config.get("model.naive_bayes.alpha"))
        1.0
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict

    @classmethod
    def load(cls, config_path: str) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse config file: {e}")

        return cls(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config instance
        """
        return cls(config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Supports nested keys using dot notation (e.g., "model.naive_bayes.alpha").

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default

        Example:
            >>> config = Config.from_dict({"model": {"alpha": 1.0}})
            >>> print(config.get("model.alpha"))
            1.0
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_env(
        self, key: str, default: Optional[str] = None
    ) -> Optional[str]:
        """
        Get value from environment variable.

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Environment variable value or default
        """
        return os.getenv(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"Config({self._config})"


# Default configuration
DEFAULT_CONFIG = {
    'model': {
        'naive_bayes': {
            'alpha': 1.0
        }
    },
    'preprocessing': {
        'lowercase': True,
        'remove_stopwords': True,
        'apply_stemming': True,
        'remove_punctuation': True
    },
    'training': {
        'test_size': 0.2,
        'random_state': 42,
        'stratify': True
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    },
    'paths': {
        'data': 'data/train.csv',
        'models': 'data/models',
        'logs': 'logs'
    }
}


def get_default_config() -> Config:
    """
    Get default configuration.

    Returns:
        Default Config instance
    """
    return Config.from_dict(DEFAULT_CONFIG)
