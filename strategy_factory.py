# strategies/strategy_factory.py
import logging
from typing import Dict, Any, List

from config.app_config import AppConfig
from risk.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy
from strategies.scalping import ScalpingStrategy
from strategies.grid_trading import GridTradingStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.sentiment_based import SentimentBasedStrategy
from strategies.arbitrage import ArbitrageStrategy

logger = logging.getLogger(__name__)

class StrategyFactory:
    """
    Factory for creating strategy instances
    """
    def __init__(self, config: AppConfig, risk_manager: RiskManager):
        self.config = config
        self.risk_manager = risk_manager
        
        # Register strategies
        self.strategies = {
            'scalping': ScalpingStrategy,
            'grid_trading': GridTradingStrategy,
            'trend_following': TrendFollowingStrategy,
            'mean_reversion': MeanReversionStrategy,
            'sentiment_based': SentimentBasedStrategy,
            'arbitrage': ArbitrageStrategy
        }
        
    def create_strategy(self, strategy_name: str, exchange, user, allocation_percent: float = 10) -> BaseStrategy:
        """
        Create a strategy instance
        
        Args:
            strategy_name: Name of the strategy
            exchange: Exchange instance
            user: User instance
            allocation_percent: Percentage of capital to allocate
            
        Returns:
            BaseStrategy: Strategy instance
        """
        try:
            # Get strategy class
            strategy_class = self.strategies.get(strategy_name.lower())
            
            if not strategy_class:
                logger.error(f"Unsupported strategy: {strategy_name}")
                raise ValueError(f"Unsupported strategy: {strategy_name}")
                
            # Get strategy configuration
            strategy_config = self.config.get(f"strategies.{strategy_name}", {})
            
            # Add allocation percentage
            strategy_config['allocation_percent'] = allocation_percent
            
            # Add trading pairs
            if user and hasattr(user, 'trading_pairs'):
                strategy_config['symbols'] = [
                    pair for pair, enabled in user.trading_pairs.items() if enabled
                ]
            else:
                strategy_config['symbols'] = self.config.get('trading.default_pairs', [])
                
            # Create strategy instance
            strategy = strategy_class(
                exchange=exchange,
                user=user,
                risk_manager=self.risk_manager,
                config=strategy_config
            )
            
            logger.info(f"Created {strategy_name} strategy instance with {allocation_percent}% allocation")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error creating strategy instance: {str(e)}")
            raise
            
    def get_available_strategies(self) -> List[str]:
        """
        Get list of available strategies
        
        Returns:
            List[str]: List of strategy names
        """
        return list(self.strategies.keys())