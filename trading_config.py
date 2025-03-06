# config/trading_config.py
from typing import Dict, List, Any
import logging
import os
import json
from config.app_config import AppConfig

logger = logging.getLogger(__name__)

class TradingConfig:
    """
    Trading-specific configuration
    """
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """
        Get singleton instance
        
        Returns:
            TradingConfig: Singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self, config_path: str = "config/settings.json"):
        # Get AppConfig instance
        self.app_config = None
        if isinstance(config_path, AppConfig):
            self.app_config = config_path
        else:
            self.app_config = AppConfig(config_path)
            
        self.trading_pairs = {}
        self.timeframes = []
        self.strategy_configs = {}
        self.load_config()
        
    def load_config(self) -> None:
        """
        Load trading configuration
        """
        try:
            # Load trading pairs
            self.trading_pairs = self.app_config.get("trading.pairs", {})
            if not self.trading_pairs:
                default_pairs = self.app_config.get("trading.default_pairs", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
                self.trading_pairs = {pair: True for pair in default_pairs}
                
            # Load timeframes
            self.timeframes = self.app_config.get("trading.timeframes", [])
            if not self.timeframes:
                self.timeframes = self.app_config.get("trading.default_timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"])
                
            # Load strategy configurations
            self.strategy_configs = self.app_config.get("strategies", {})
            if not self.strategy_configs:
                self.strategy_configs = self._get_default_strategy_configs()
                
            logger.info("Trading configuration loaded")
                
        except Exception as e:
            logger.error(f"Error loading trading configuration: {str(e)}")
            
    def _get_default_strategy_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get default strategy configurations
        """
        return {
            "scalping": {
                "enabled": True,
                "timeframes": ["1m", "5m"],
                "parameters": {
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30,
                    "ema_fast_period": 9,
                    "ema_slow_period": 21,
                    "take_profit_percent": 1.0,
                    "stop_loss_percent": 0.5
                }
            },
            "grid_trading": {
                "enabled": True,
                "timeframes": ["1h"],
                "parameters": {
                    "grid_levels": 10,
                    "grid_spacing_percent": 1.0,
                    "total_investment_percent": 20.0,
                    "dynamic_boundaries": True,
                    "volatility_factor": 1.5
                }
            },
            "trend_following": {
                "enabled": True,
                "timeframes": ["4h", "1d"],
                "parameters": {
                    "ema_short_period": 20,
                    "ema_long_period": 50,
                    "macd_fast_period": 12,
                    "macd_slow_period": 26,
                    "macd_signal_period": 9,
                    "take_profit_percent": 5.0,
                    "stop_loss_percent": 2.0,
                    "trailing_stop_percent": 1.0
                }
            },
            "mean_reversion": {
                "enabled": True,
                "timeframes": ["15m", "1h"],
                "parameters": {
                    "bollinger_period": 20,
                    "bollinger_std_dev": 2.0,
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30,
                    "take_profit_percent": 2.0,
                    "stop_loss_percent": 1.0
                }
            },
            "sentiment_based": {
                "enabled": True,
                "timeframes": ["1h", "4h"],
                "parameters": {
                    "sentiment_threshold_positive": 0.6,
                    "sentiment_threshold_negative": 0.4,
                    "news_impact_time_hours": 24,
                    "volatility_adjustment": True,
                    "take_profit_percent": 3.0,
                    "stop_loss_percent": 2.0
                }
            },
            "arbitrage": {
                "enabled": True,
                "parameters": {
                    "min_profit_percent": 0.5,
                    "max_execution_time_ms": 1000,
                    "max_slippage_percent": 0.1,
                    "triangular": {
                        "enabled": True,
                        "min_profit_percent": 0.2
                    },
                    "cross_exchange": {
                        "enabled": True,
                        "exchanges": ["binance", "kucoin"],
                        "min_profit_percent": 0.8
                    }
                }
            }
        }
        
    def get_enabled_pairs(self) -> List[str]:
        """
        Get list of enabled trading pairs
        """
        return [pair for pair, enabled in self.trading_pairs.items() if enabled]
        
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific strategy
        """
        return self.strategy_configs.get(strategy_name, {})
        
    def get_strategy_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific strategy
        """
        strategy_config = self.get_strategy_config(strategy_name)
        return strategy_config.get("parameters", {})
        
    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """
        Check if a strategy is enabled
        """
        strategy_config = self.get_strategy_config(strategy_name)
        return strategy_config.get("enabled", False)
        
    def get_strategy_timeframes(self, strategy_name: str) -> List[str]:
        """
        Get timeframes for a specific strategy
        """
        strategy_config = self.get_strategy_config(strategy_name)
        return strategy_config.get("timeframes", [])
        
    def save_strategy_config(self, strategy_name: str, config: Dict[str, Any]) -> None:
        """
        Save configuration for a specific strategy
        """
        try:
            self.strategy_configs[strategy_name] = config
            # Use app_config's set method if it exists
            if hasattr(self.app_config, 'set'):
                self.app_config.set(f"strategies.{strategy_name}", config)
                if hasattr(self.app_config, 'save_config'):
                    self.app_config.save_config()
            logger.info(f"Saved configuration for strategy: {strategy_name}")
        except Exception as e:
            logger.error(f"Error saving strategy configuration: {str(e)}")
            
    def enable_strategy(self, strategy_name: str) -> None:
        """
        Enable a strategy
        """
        if strategy_name in self.strategy_configs:
            self.strategy_configs[strategy_name]["enabled"] = True
            self.save_strategy_config(strategy_name, self.strategy_configs[strategy_name])
            logger.info(f"Enabled strategy: {strategy_name}")
        else:
            logger.warning(f"Strategy not found: {strategy_name}")
            
    def disable_strategy(self, strategy_name: str) -> None:
        """
        Disable a strategy
        """
        if strategy_name in self.strategy_configs:
            self.strategy_configs[strategy_name]["enabled"] = False
            self.save_strategy_config(strategy_name, self.strategy_configs[strategy_name])
            logger.info(f"Disabled strategy: {strategy_name}")
        else:
            logger.warning(f"Strategy not found: {strategy_name}")
            
    def update_strategy_parameters(self, strategy_name: str, parameters: Dict[str, Any]) -> None:
        """
        Update parameters for a specific strategy
        """
        if strategy_name in self.strategy_configs:
            self.strategy_configs[strategy_name]["parameters"].update(parameters)
            self.save_strategy_config(strategy_name, self.strategy_configs[strategy_name])
            logger.info(f"Updated parameters for strategy: {strategy_name}")
        else:
            logger.warning(f"Strategy not found: {strategy_name}")
            
    def enable_trading_pair(self, pair: str) -> None:
        """
        Enable a trading pair
        """
        self.trading_pairs[pair] = True
        # Use app_config's set method if it exists
        if hasattr(self.app_config, 'set'):
            self.app_config.set(f"trading.pairs.{pair}", True)
            if hasattr(self.app_config, 'save_config'):
                self.app_config.save_config()
        logger.info(f"Enabled trading pair: {pair}")
        
    def disable_trading_pair(self, pair: str) -> None:
        """
        Disable a trading pair
        """
        if pair in self.trading_pairs:
            self.trading_pairs[pair] = False
            # Use app_config's set method if it exists
            if hasattr(self.app_config, 'set'):
                self.app_config.set(f"trading.pairs.{pair}", False)
                if hasattr(self.app_config, 'save_config'):
                    self.app_config.save_config()
            logger.info(f"Disabled trading pair: {pair}")
        else:
            logger.warning(f"Trading pair not found: {pair}")
    
    def get_all(self):
        """
        Get all trading configuration settings
        
        Returns:
            Dict: All trading configuration settings
        """
        return {
            "trading_pairs": self.trading_pairs,
            "timeframes": self.timeframes,
            "strategies": self.strategy_configs
        }