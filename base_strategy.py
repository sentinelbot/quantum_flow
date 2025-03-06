# strategies/base_strategy.py
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import time

from exchange.abstract_exchange import AbstractExchange, TradeSignal

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Base strategy class that all strategies must inherit from
    """
    def __init__(self, name: str, exchange: AbstractExchange, user, risk_manager, config: Dict[str, Any]):
        self.name = name
        self.exchange = exchange
        self.user = user
        self.risk_manager = risk_manager
        self.config = config
        self.enabled = True
        self.last_update_time = 0
        self.update_interval = 60  # seconds
        self.timeframes = config.get('timeframes', [])
        self.symbols = config.get('symbols', [])
        self.allocation_percent = config.get('allocation_percent', 10)
        
        # Data cache
        self.market_data = {}
        
    def is_enabled(self) -> bool:
        """
        Check if strategy is enabled
        """
        return self.enabled
        
    def enable(self) -> None:
        """
        Enable strategy
        """
        self.enabled = True
        logger.info(f"Strategy {self.name} enabled")
        
    def disable(self) -> None:
        """
        Disable strategy
        """
        self.enabled = False
        logger.info(f"Strategy {self.name} disabled")
        
    def update(self) -> None:
        """
        Update strategy with latest market data
        """
        current_time = time.time()
        
        # Check if update is needed
        if current_time - self.last_update_time < self.update_interval:
            return
            
        try:
            # Update market data for all symbols and timeframes
            for symbol in self.symbols:
                self.market_data[symbol] = {}
                
                for timeframe in self.timeframes:
                    candles = self.exchange.get_historical_data(symbol, timeframe, limit=100)
                    self.market_data[symbol][timeframe] = candles
                    
            # Custom strategy update
            self._update()
            
            # Update last update time
            self.last_update_time = current_time
            
        except Exception as e:
            logger.error(f"Error updating strategy {self.name}: {str(e)}")
            
    @abstractmethod
    def _update(self) -> None:
        """
        Custom strategy update logic
        To be implemented by subclasses
        """
        pass
        
    def generate_signals(self) -> List[TradeSignal]:
        """
        Generate trading signals
        """
        if not self.is_enabled():
            return []
            
        try:
            # Generate signals
            signals = self._generate_signals()
            
            # Log signals
            if signals:
                logger.info(f"Strategy {self.name} generated {len(signals)} signals")
                
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for strategy {self.name}: {str(e)}")
            return []
            
    @abstractmethod
    def _generate_signals(self) -> List[TradeSignal]:
        """
        Custom signal generation logic
        To be implemented by subclasses
        """
        pass
        
    def get_candles(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """
        Get cached candles for a symbol and timeframe
        """
        return self.market_data.get(symbol, {}).get(timeframe, [])
        
    def calculate_take_profit_price(self, entry_price: float, side: str) -> float:
        """
        Calculate take profit price
        """
        take_profit_percent = self.config.get('parameters', {}).get('take_profit_percent', 1.0)
        
        if side.lower() == 'buy':
            return entry_price * (1 + take_profit_percent / 100)
        else:
            return entry_price * (1 - take_profit_percent / 100)
            
    def calculate_stop_loss_price(self, entry_price: float, side: str) -> float:
        """
        Calculate stop loss price
        """
        stop_loss_percent = self.config.get('parameters', {}).get('stop_loss_percent', 0.5)
        
        if side.lower() == 'buy':
            return entry_price * (1 - stop_loss_percent / 100)
        else:
            return entry_price * (1 + stop_loss_percent / 100)