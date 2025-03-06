import logging
import threading
import time
from typing import Dict, List, Optional, Any, Set, Tuple

from config.app_config import AppConfig
from config.trading_config import TradingConfig
from database.db import DatabaseManager
from exchange.abstract_exchange import AbstractExchange
from exchange.exchange_factory import ExchangeFactory
from notification.notification_manager import NotificationManager
from risk.risk_manager import RiskManager
from security.api_key_manager import APIKeyManager
from strategies.strategy_factory import StrategyFactory
from strategies.base_strategy import BaseStrategy
from analysis.market_analyzer import MarketAnalyzer
from ml.models.price_predictor import PricePredictor
from ml.models.pattern_recognition import PatternRecognition
from ml.models.regime_classifier import RegimeClassifier
from database.repository.user_repository import UserRepository
from database.repository.trade_repository import TradeRepository
from database.repository.position_repository import PositionRepository
from profit.profit_tracker import ProfitTracker

logger = logging.getLogger(__name__)

class TradingEngine:
    """
    Comprehensive Trading Execution and Management System

    Provides an advanced, configurable platform for multi-user, 
    multi-strategy trading across various exchanges with sophisticated 
    risk management and performance tracking capabilities.
    """

    def __init__(
        self, 
        config: Optional[AppConfig] = None,
        trading_config: Optional[TradingConfig] = None,
        db: Optional[DatabaseManager] = None,
        exchange_factory: Optional[ExchangeFactory] = None,
        api_key_manager: Optional[APIKeyManager] = None,
        notification_manager: Optional[NotificationManager] = None,
        risk_manager: Optional[RiskManager] = None,
        strategy_factory: Optional[StrategyFactory] = None,
        market_analyzer: Optional[MarketAnalyzer] = None,
        price_predictor: Optional[PricePredictor] = None,
        pattern_recognition: Optional[PatternRecognition] = None,
        regime_classifier: Optional[RegimeClassifier] = None,
        position_repository: Optional[PositionRepository] = None,
        trade_repository: Optional[TradeRepository] = None,
        user_repository: Optional[UserRepository] = None,
        profit_tracker: Optional[ProfitTracker] = None,
        shutdown_event: Optional[threading.Event] = None
    ):
        """
        Initialize the Trading Engine with comprehensive configuration.

        Args:
            config (AppConfig): Application configuration
            trading_config (TradingConfig): Trading-specific configuration
            db (DatabaseManager): Database management system
            exchange_factory (ExchangeFactory): Factory for creating exchange connections
            api_key_manager (APIKeyManager): Manager for handling API keys
            notification_manager (NotificationManager): System for sending notifications
            risk_manager (RiskManager): Risk management and validation system
            strategy_factory (StrategyFactory): Factory for creating trading strategies
            market_analyzer (MarketAnalyzer): Market analysis system
            price_predictor (PricePredictor): Price prediction model
            pattern_recognition (PatternRecognition): Pattern recognition model
            regime_classifier (RegimeClassifier): Market regime classification model
            position_repository (PositionRepository): Repository for tracking positions
            trade_repository (TradeRepository): Repository for tracking trades
            user_repository (UserRepository): Repository for user management
            profit_tracker (ProfitTracker): System for tracking trading profits
            shutdown_event (threading.Event): Event to signal system shutdown
        """
        # Core system dependencies
        self.config = config
        self.trading_config = trading_config
        self.db = db
        self.exchange_factory = exchange_factory
        self.api_key_manager = api_key_manager
        self.notification_manager = notification_manager
        self.shutdown_event = shutdown_event or threading.Event()

        # Risk and strategy management
        self.risk_manager = risk_manager or RiskManager(config, db)
        self.strategy_factory = strategy_factory or StrategyFactory(config, self.risk_manager)

        # Analysis and ML components
        self.market_analyzer = market_analyzer
        self.price_predictor = price_predictor
        self.pattern_recognition = pattern_recognition
        self.regime_classifier = regime_classifier

        # Repositories - direct references, not using db.get_repository()
        self.position_repository = position_repository
        self.trade_repository = trade_repository
        self.user_repository = user_repository

        # Profit tracking
        self.profit_tracker = profit_tracker

        # User tracking
        self.user_exchange_map: Dict[int, AbstractExchange] = {}
        self.user_strategy_map: Dict[int, List[BaseStrategy]] = {}
        self.active_users_cache: List[Any] = []
        self.last_user_refresh = 0
        self.user_refresh_interval = 300  # 5 minutes

        # Trading thread management
        self.trading_thread: Optional[threading.Thread] = None
        self.running = False
        self.paused = False

        # Market data cache
        self.market_data_cache = {}
        self.last_market_update = 0
        self.market_update_interval = 60  # 1 minute

        # Error tracking
        self.error_counts = {}
        self.max_consecutive_errors = 5
        self.error_backoff_time = 5.0  # seconds

        # Health status
        self.health_status = "initializing"
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Schema verification flag
        self.schema_verified = False

        logger.info("Trading Engine initialized successfully")
        
        # Verify database schema for critical tables
        self._verify_database_schema()

    def _verify_database_schema(self) -> None:
        """
        Verify that the database schema has necessary columns.
        
        This is a preliminary check to identify schema issues early.
        """
        if not self.db or self.schema_verified:
            return
            
        try:
            # Check if the db has schema verification capabilities
            if hasattr(self.db, 'has_column'):
                # Check for critical columns in users table
                users_table_status = {
                    'is_admin': self.db.has_column('users', 'is_admin'),
                    'is_active': self.db.has_column('users', 'is_active'),
                    'is_paused': self.db.has_column('users', 'is_paused')
                }
                
                for col_name, exists in users_table_status.items():
                    if not exists:
                        logger.warning(f"Critical column '{col_name}' missing from users table")
                
                self.schema_verified = True
                
        except Exception as e:
            logger.error(f"Error verifying database schema: {str(e)}")

    def start(self) -> bool:
        """
        Activate the trading engine and begin trade execution.

        Initiates a background thread for continuous trading operations 
        and user strategy management.

        Returns:
            bool: Indicates successful engine start
        """
        with self.lock:
            if self.running:
                logger.warning("Trading engine is already operational")
                return False

            logger.info("Starting trading engine")
            self.running = True
            self.paused = False
            self.health_status = "starting"
            self.trading_thread = threading.Thread(target=self._trading_loop, name="trading_engine_thread", daemon=True)
            self.trading_thread.start()

            logger.info("Trading engine started successfully")
            self.health_status = "running"
            return True

    def stop(self) -> bool:
        """
        Gracefully terminate the trading engine.

        Closes all exchange connections and stops active trading processes.

        Returns:
            bool: Indicates successful engine shutdown
        """
        with self.lock:
            if not self.running:
                logger.warning("Trading engine is already inactive")
                return False

            logger.info("Stopping trading engine")
            self.running = False
            self.health_status = "stopping"

            # Wait for trading thread to terminate
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=30)
                if self.trading_thread.is_alive():
                    logger.warning("Trading thread did not terminate gracefully")

            # Close exchange connections
            self._close_exchange_connections()

            logger.info("Trading engine stopped successfully")
            self.health_status = "stopped"
            return True

    def pause(self) -> bool:
        """
        Pause trading operations without shutting down.
        
        Temporarily suspends trading activities while maintaining
        exchange connections and system state.
        
        Returns:
            bool: Indicates successful pause
        """
        with self.lock:
            if not self.running:
                logger.warning("Trading engine is not running")
                return False
                
            if self.paused:
                logger.warning("Trading engine is already paused")
                return False
                
            logger.info("Pausing trading engine")
            self.paused = True
            self.health_status = "paused"
            return True
            
    def resume(self) -> bool:
        """
        Resume trading operations after a pause.
        
        Restarts trading activities using the existing
        exchange connections and system state.
        
        Returns:
            bool: Indicates successful resume
        """
        with self.lock:
            if not self.running:
                logger.warning("Trading engine is not running")
                return False
                
            if not self.paused:
                logger.warning("Trading engine is not paused")
                return False
                
            logger.info("Resuming trading engine")
            self.paused = False
            self.health_status = "running"
            return True

    def update_market_data(self):
        """
        Refresh market data for all active trading symbols.
        
        Fetches latest price, order book, and trading volume data 
        to ensure trading decisions are made with current information.
        
        Returns:
            bool: True if update was successful
        """
        try:
            logger.info("Updating market data for all active symbols")
            
            # Get all active symbols being traded
            active_symbols = self._get_active_trading_symbols()
            
            if not active_symbols:
                logger.info("No active trading symbols found")
                return True
                
            # Track success count for reporting
            success_count = 0
            
            # Update market data for each exchange instance
            for user_id, exchange in self.user_exchange_map.items():
                try:
                    # Get user's active symbols
                    user_symbols = [s for s in active_symbols if s.get('user_id') == user_id]
                    
                    if not user_symbols:
                        continue
                        
                    # Update market data for this user's exchange
                    for symbol_data in user_symbols:
                        symbol = symbol_data.get('symbol')
                        # Store market data in cache
                        market_data = exchange.get_market_data(symbol)
                        if market_data:
                            key = f"{user_id}:{symbol}"
                            self.market_data_cache[key] = {
                                'data': market_data,
                                'timestamp': time.time()
                            }
                            success_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to update market data for user {user_id}: {str(e)}")
            
            # Update last market update timestamp
            self.last_market_update = time.time()
            
            logger.info(f"Market data updated successfully for {success_count} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Market data update failed: {str(e)}")
            return False

    def analyze_performance(self):
        """
        Analyze trading performance across all strategies and users.
        
        Generates performance metrics including profit/loss, drawdown,
        win rate, and returns on investment.
        
        Returns:
            bool: True if analysis was successful
        """
        try:
            logger.info("Analyzing trading system performance")
            
            if not self.profit_tracker:
                logger.warning("Profit tracker not available, skipping performance analysis")
                return False
            
            # Generate performance reports
            self.profit_tracker.generate_performance_report()
            
            # Log summary statistics
            summary = self.profit_tracker.get_summary_statistics()
            logger.info(f"Performance analysis completed: {summary}")
            
            return True
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {str(e)}")
            return False

    def evaluate_strategies(self):
        """
        Evaluate the effectiveness of trading strategies.
        
        Reviews historical performance of each strategy type and
        suggests optimization opportunities.
        
        Returns:
            bool: True if evaluation was successful
        """
        try:
            logger.info("Evaluating trading strategies performance")
            
            if not self.trade_repository:
                logger.warning("Trade repository not available, skipping strategy evaluation")
                return False
            
            # Get strategy performance metrics
            strategy_metrics = self._get_strategy_performance_metrics()
            
            # Log strategy performance
            for strategy_name, metrics in strategy_metrics.items():
                logger.info(f"Strategy evaluation - {strategy_name}: win_rate={metrics.get('win_rate', 0):.2f}%, " +
                           f"avg_profit={metrics.get('avg_profit', 0):.2f}%, " +
                           f"trades={metrics.get('trade_count', 0)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Strategy evaluation failed: {str(e)}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the trading engine.
        
        Provides comprehensive information about the trading engine's
        state, active users, strategies, and performance.
        
        Returns:
            Dict with detailed status information
        """
        with self.lock:
            return {
                "running": self.running,
                "paused": self.paused,
                "health_status": self.health_status,
                "active_users_count": len(self.active_users_cache),
                "active_exchanges_count": len(self.user_exchange_map),
                "active_strategies_count": sum(len(strat_list) for strat_list in self.user_strategy_map.values()),
                "market_data_age_seconds": time.time() - self.last_market_update if self.last_market_update else None,
                "user_data_age_seconds": time.time() - self.last_user_refresh if self.last_user_refresh else None,
                "last_error": self._get_last_error(),
                "schema_verified": self.schema_verified
            }

    def _get_last_error(self) -> Optional[str]:
        """
        Get the most recent error message.
        
        Returns:
            Most recent error message or None
        """
        if not self.error_counts:
            return None
            
        max_count_key = max(self.error_counts.items(), key=lambda x: x[1][0])
        return max_count_key[0]

    def _get_strategy_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for each strategy.
        
        Returns:
            Dict: Strategy performance metrics by strategy name
        """
        try:
            metrics = {}
            
            # Get all trades from the past 30 days
            recent_trades = self.trade_repository.get_trades_since_days(30)
            
            # Group trades by strategy
            strategy_trades = {}
            for trade in recent_trades:
                strategy_name = trade.strategy_name
                if strategy_name not in strategy_trades:
                    strategy_trades[strategy_name] = []
                strategy_trades[strategy_name].append(trade)
            
            # Calculate metrics for each strategy
            for strategy_name, trades in strategy_trades.items():
                win_count = sum(1 for trade in trades if trade.profit_pct > 0)
                total_count = len(trades)
                win_rate = (win_count / total_count * 100) if total_count > 0 else 0
                
                profits = [trade.profit_pct for trade in trades]
                avg_profit = sum(profits) / len(profits) if profits else 0
                
                metrics[strategy_name] = {
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'trade_count': total_count
                }
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating strategy metrics: {str(e)}")
            return {}

    def _get_active_trading_symbols(self) -> List[Dict[str, Any]]:
        """
        Get all symbols currently being traded across all active users.
        
        Returns:
            list: List of active trading symbols with metadata
        """
        try:
            active_symbols = []
            
            # If we have direct access to position repository
            if self.position_repository:
                try:
                    # Get all active positions
                    active_positions = self.position_repository.get_all_active_positions()
                    
                    # Extract symbols from positions
                    for position in active_positions:
                        active_symbols.append({
                            'user_id': position.user_id,
                            'symbol': position.symbol,
                            'exchange': position.exchange
                        })
                    
                    if active_symbols:
                        return active_symbols
                except Exception as e:
                    logger.error(f"Error getting positions from repository: {str(e)}")
                    # Continue with fallback method
            
            # Fallback method: collect from user strategies
            for user_id, strategies in self.user_strategy_map.items():
                for strategy in strategies:
                    if hasattr(strategy, 'get_trading_symbols'):
                        try:
                            symbols = strategy.get_trading_symbols()
                            for symbol in symbols:
                                active_symbols.append({
                                    'user_id': user_id,
                                    'symbol': symbol,
                                    'exchange': self.user_exchange_map.get(user_id).name if user_id in self.user_exchange_map else None
                                })
                        except Exception as e:
                            logger.error(f"Error getting symbols from strategy: {str(e)}")
            
            # Remove duplicates
            unique_symbols = []
            seen = set()
            
            for item in active_symbols:
                key = f"{item['user_id']}:{item['symbol']}"
                if key not in seen:
                    seen.add(key)
                    unique_symbols.append(item)
                    
            return unique_symbols
            
        except Exception as e:
            logger.error(f"Failed to get active trading symbols: {str(e)}")
            return []

    def _trading_loop(self) -> None:
        """
        Primary trading execution loop.

        Continuously processes user strategies and manages trading activities 
        while respecting system shutdown signals.
        """
        error_count = 0
        last_market_update = 0
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Skip processing if paused
                if self.paused:
                    time.sleep(1)
                    continue
                    
                # Update market data periodically
                current_time = time.time()
                if current_time - last_market_update > self.market_update_interval:
                    self.update_market_data()
                    last_market_update = current_time
                
                # Load and process active users
                active_users = self._load_active_users()
                
                for user in active_users:
                    self._process_user_strategies(user)
                
                # Reset error count on successful iteration
                error_count = 0
                
                # Prevent high CPU usage
                time.sleep(1)
            
            except Exception as e:
                error_count += 1
                error_message = str(e)
                
                # Track error frequency
                if error_message not in self.error_counts:
                    self.error_counts[error_message] = [0, time.time()]
                self.error_counts[error_message][0] += 1
                self.error_counts[error_message][1] = time.time()
                
                logger.error(f"Critical error in trading loop: {error_message}")
                
                # Implement exponential backoff for repeated errors
                backoff_time = min(self.error_backoff_time * (2 ** (error_count - 1)), 60)
                time.sleep(backoff_time)
                
                # If too many consecutive errors, mark health as degraded
                if error_count >= self.max_consecutive_errors:
                    self.health_status = "degraded"
                    logger.critical(f"Trading engine health degraded after {error_count} consecutive errors")

    def _load_active_users(self) -> List[Any]:
        """
        Retrieve and initialize active users for trading with schema adaptation.
        
        Implements caching to reduce database load and handles repository errors.
        Uses multiple fallback mechanisms to handle database schema issues.

        Returns:
            List of active users ready for strategy processing
        """
        try:
            # Only refresh user list periodically
            current_time = time.time()
            if current_time - self.last_user_refresh < self.user_refresh_interval and self.active_users_cache:
                return self.active_users_cache
                
            active_users = []
            
            # Use user repository directly with schema-adaptive approach
            if self.user_repository:
                # Check if repository has schema-adaptive methods
                if hasattr(self.user_repository, 'get_active_users'):
                    try:
                        active_users = self.user_repository.get_active_users()
                    except Exception as e:
                        # Handle schema-related errors
                        if "column users.is_admin does not exist" in str(e):
                            logger.warning("Using fallback method for active users due to schema issue")
                            active_users = self._get_active_users_fallback()
                        else:
                            logger.error(f"Error getting active users: {str(e)}")
                            # Use cached users if available
                            return self.active_users_cache or []
                else:
                    # Old repository without enhanced methods
                    active_users = self._get_active_users_fallback()
            else:
                # Fallback to database if direct repository not available
                try:
                    user_repo = self.db.get_repository('user')
                    active_users = user_repo.get_active_users()
                except Exception as e:
                    logger.error(f"Failed to get user repository: {str(e)}")
                    # Try raw query as last resort
                    active_users = self._get_active_users_fallback()
            
            if not active_users and self.active_users_cache:
                # If we failed to get users but have a cache, use it
                logger.warning("Failed to get active users, using cached data")
                return self.active_users_cache
            
            # Update cache
            self.active_users_cache = active_users
            self.last_user_refresh = current_time

            # Initialize exchange connections for new users
            for user in active_users:
                if user.id not in self.user_exchange_map:
                    self._initialize_user(user)

            return active_users
        
        except Exception as e:
            logger.error(f"Error loading active users: {str(e)}")
            # Return cached users if available, otherwise empty list
            return self.active_users_cache or []

    def _get_active_users_fallback(self) -> List[Any]:
        """
        Fallback method to get active users without relying on ORM.
        
        Uses raw SQL queries to bypass potential schema issues.
        
        Returns:
            List of User objects with is_active=True
        """
        from database.models.user import User  # Import here to avoid circular imports
        
        if not self.db:
            return []
            
        try:
            # Try direct SQL query with only essential fields
            users = []
            
            try:
                # Use database's execute_query method if available
                if hasattr(self.db, 'execute_query'):
                    result = self.db.execute_query("SELECT id, telegram_id, email, username, first_name, last_name FROM users WHERE is_active = true")
                    
                    # Convert results to User objects
                    for row in result:
                        user = User(
                            id=row.get('id'),
                            telegram_id=row.get('telegram_id'),
                            email=row.get('email'),
                            username=row.get('username'),
                            first_name=row.get('first_name'),
                            last_name=row.get('last_name'),
                            is_active=True,
                            is_admin=False  # Default value for missing column
                        )
                        users.append(user)
                    
                    return users
            except Exception as e:
                logger.error(f"Error in fallback query: {str(e)}")
                
            # Last resort - use SQLAlchemy session directly
            try:
                session = self.db.get_session()
                try:
                    from sqlalchemy import text
                    result = session.execute(text("SELECT id, telegram_id, email, username FROM users WHERE is_active = true"))
                    
                    for row in result:
                        # Manual mapping of result rows to User objects
                        user = User(
                            id=row[0],
                            telegram_id=row[1],
                            email=row[2],
                            username=row[3],
                            is_active=True,
                            is_admin=False
                        )
                        users.append(user)
                        
                    return users
                except Exception as inner_e:
                    logger.error(f"Error in SQLAlchemy direct query: {str(inner_e)}")
                    return []
                finally:
                    session.close()
            except Exception as session_e:
                logger.error(f"Error getting session: {str(session_e)}")
                return []
                
        except Exception as e:
            logger.error(f"All fallback methods failed: {str(e)}")
            return []

    def _initialize_user(self, user) -> bool:
        """
        Establish exchange connection and strategies for a user.

        Args:
            user: User object to initialize
            
        Returns:
            bool: Success status
        """
        try:
            # Skip if user already initialized
            if user.id in self.user_exchange_map and user.id in self.user_strategy_map:
                return True
                
            # Retrieve API keys
            if not self.api_key_manager:
                logger.warning(f"API key manager not available, skipping user {user.id} initialization")
                return False
                
            api_keys = self.api_key_manager.get_api_keys(user.id)
            if not api_keys:
                logger.warning(f"No API keys found for user {user.id}")
                return False
            
            # Check for required exchange factory
            if not self.exchange_factory:
                logger.warning(f"Exchange factory not available, skipping user {user.id} initialization")
                return False

            # Create and initialize exchange
            exchange_name = getattr(user, 'preferred_exchange', self.trading_config.get('default_exchange', 'binance'))
            exchange = self.exchange_factory.create_exchange(
                exchange_name,
                api_keys['api_key'],
                api_keys['secret_key']
            )
            
            # Initialize exchange connection
            exchange_initialized = exchange.initialize()
            if not exchange_initialized:
                logger.warning(f"Failed to initialize exchange for user {user.id}")
                return False

            # Store exchange and initialize strategies
            self.user_exchange_map[user.id] = exchange
            self._initialize_user_strategies(user)

            logger.info(f"Initialized user {user.id} trading environment")
            return True
        
        except Exception as e:
            logger.error(f"User initialization failed for user {user.id}: {str(e)}")
            return False

    def _close_exchange_connections(self):
        """
        Safely close all active exchange connections.
        """
        for user_id, exchange in list(self.user_exchange_map.items()):
            try:
                exchange.close()
                logger.info(f"Closed exchange connection for user {user_id}")
            except Exception as e:
                logger.error(f"Exchange connection close error for user {user_id}: {str(e)}")
        
        self.user_exchange_map.clear()
        self.user_strategy_map.clear()

    def _initialize_user_strategies(self, user) -> bool:
        """
        Configure trading strategies for a specific user.

        Args:
            user: User object to configure strategies for
            
        Returns:
            bool: Success status
        """
        try:
            # Check if strategy factory is available
            if not self.strategy_factory:
                logger.warning(f"Strategy factory not available, skipping strategies for user {user.id}")
                return False
                
            # Skip if user has no exchange
            if user.id not in self.user_exchange_map:
                logger.warning(f"No exchange found for user {user.id}, skipping strategy initialization")
                return False
                
            # Get user's strategy configuration
            strategy_config = None
            
            # Try to get from user repository first
            if self.user_repository:
                try:
                    if hasattr(self.user_repository, 'get_strategy_config'):
                        strategy_config = self.user_repository.get_strategy_config(user.id)
                    else:
                        logger.warning(f"User repository lacks get_strategy_config method")
                except Exception as e:
                    logger.error(f"Error getting strategy config from repository: {str(e)}")
            
            # Fallback to directly accessing user object
            if not strategy_config and hasattr(user, 'strategy_config'):
                strategy_config = user.strategy_config
            
            # Fallback to database if needed
            if not strategy_config and self.db:
                try:
                    user_repo = self.db.get_repository('user')
                    if hasattr(user_repo, 'get_strategy_config'):
                        strategy_config = user_repo.get_strategy_config(user.id)
                except Exception as e:
                    logger.error(f"Error getting strategy config from database: {str(e)}")
            
            # Use default config if still no user-specific configuration
            if not strategy_config:
                # Get risk level from user object if available, otherwise default to 'medium'
                risk_level = getattr(user, 'risk_level', None)
                if risk_level and hasattr(risk_level, 'value'):
                    # Handle enum case
                    risk_level = risk_level.value
                strategy_config = self._get_default_strategy_config(risk_level or 'medium')

            # Create strategy instances
            strategies = []
            for strategy_name, allocation in strategy_config.items():
                if allocation > 0:
                    strategy = self.strategy_factory.create_strategy(
                        strategy_name,
                        self.user_exchange_map[user.id],
                        user,
                        allocation
                    )
                    if strategy:
                        strategies.append(strategy)

            # Store user strategies
            self.user_strategy_map[user.id] = strategies
            logger.info(f"Initialized {len(strategies)} strategies for user {user.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize strategies for user {user.id}: {str(e)}")
            return False

    def _get_default_strategy_config(self, risk_level: str) -> Dict[str, int]:
        """
        Generate default strategy configuration based on risk profile.

        Args:
            risk_level: User's risk tolerance level

        Returns:
            Dictionary of strategy allocations
        """
        strategy_configs = {
            "low": {
                "grid_trading": 40,
                "mean_reversion": 30,
                "trend_following": 30,
                "scalping": 0,
                "sentiment_based": 0,
                "arbitrage": 0
            },
            "medium": {
                "grid_trading": 20,
                "mean_reversion": 20,
                "trend_following": 30,
                "scalping": 20,
                "sentiment_based": 10,
                "arbitrage": 0
            },
            "high": {
                "grid_trading": 10,
                "mean_reversion": 15,
                "trend_following": 25,
                "scalping": 30,
                "sentiment_based": 10,
                "arbitrage": 10
            }
        }

        return strategy_configs.get(risk_level.lower(), {
            "grid_trading": 25,
            "mean_reversion": 25,
            "trend_following": 25,
            "scalping": 15,
            "sentiment_based": 5,
            "arbitrage": 5
        })

    def _process_user_strategies(self, user) -> None:
        """
        Process all strategies for a specific user.

        Args:
            user: User object to process strategies for
        """
        # Handle missing is_paused attribute gracefully
        user_paused = False
        if hasattr(user, 'is_paused'):
            user_paused = user.is_paused
        
        # Skip processing for paused users or users without strategies
        if user_paused or user.id not in self.user_strategy_map:
            return

        # Retrieve user's exchange instance
        exchange = self.user_exchange_map.get(user.id)
        if not exchange:
            return

        # Process each strategy
        for strategy in self.user_strategy_map[user.id]:
            try:
                # Skip disabled strategies
                if not strategy.is_enabled():
                    continue

                # Update strategy with latest market data
                strategy.update()

                # Generate trading signals
                signals = strategy.generate_signals()

                # Process each generated signal
                for signal in signals:
                    self._process_signal(user, exchange, strategy, signal)

            except Exception as e:
                logger.error(f"Error processing strategy {strategy.name} for user {user.id}: {str(e)}")

    def _process_signal(self, user, exchange, strategy, signal) -> bool:
        """
        Process a trading signal through risk validation and execution.

        Args:
            user: User object
            exchange: Exchange instance
            strategy: Trading strategy
            signal: Generated trading signal
            
        Returns:
            bool: Success status
        """
        try:
            # Skip if risk manager not available
            if not self.risk_manager:
                logger.warning("Risk manager not available, skipping signal processing")
                return False
                
            # Validate signal through risk management
            validated_signal = self.risk_manager.validate_signal(user, signal)
            if not validated_signal:
                return False

            # Execute trade
            trade_result = exchange.execute_trade(validated_signal)
            if not trade_result:
                logger.warning(f"Trade execution failed for user {user.id}, symbol {signal.symbol}")
                return False

            # Record and notify successful trades
            self._record_trade(user, strategy, validated_signal, trade_result)
            self._send_trade_notification(user, strategy, validated_signal, trade_result)
            return True

        except Exception as e:
            logger.error(f"Error processing signal for user {user.id}: {str(e)}")
            return False

    def _record_trade(self, user, strategy, signal, trade_result) -> bool:
        """
        Record trade details in the database.

        Args:
            user: User object
            strategy: Trading strategy
            signal: Trading signal
            trade_result: Execution result
            
        Returns:
            bool: Success status
        """
        try:
            # Try using trade repository directly
            if self.trade_repository:
                try:
                    self.trade_repository.add_trade(
                        user_id=user.id,
                        strategy_name=strategy.name,
                        symbol=signal.symbol,
                        side=signal.side,
                        entry_price=trade_result.price,
                        quantity=trade_result.quantity,
                        take_profit=signal.take_profit,
                        stop_loss=signal.stop_loss,
                        trade_id=trade_result.trade_id,
                        timestamp=trade_result.timestamp
                    )
                    return True
                except Exception as e:
                    logger.error(f"Error using trade repository: {str(e)}")
                    # Continue to fallback
                
            # Fallback to database if needed
            if self.db:
                try:
                    trade_repo = self.db.get_repository('trade')
                    trade_repo.add_trade(
                        user_id=user.id,
                        strategy_name=strategy.name,
                        symbol=signal.symbol,
                        side=signal.side,
                        entry_price=trade_result.price,
                        quantity=trade_result.quantity,
                        take_profit=signal.take_profit,
                        stop_loss=signal.stop_loss,
                        trade_id=trade_result.trade_id,
                        timestamp=trade_result.timestamp
                    )
                    return True
                except Exception as e:
                    logger.error(f"Error using database trade repository: {str(e)}")
                    
            logger.warning("No trade repository available, trade not recorded")
            return False
            
        except Exception as e:
            logger.error(f"Error recording trade for user {user.id}: {str(e)}")
            return False

    def _send_trade_notification(self, user, strategy, signal, trade_result) -> bool:
        """
        Send trade execution notification.

        Args:
            user: User object
            strategy: Trading strategy
            signal: Trading signal
            trade_result: Execution result
            
        Returns:
            bool: Success status
        """
        try:
            # Skip if notification manager not available
            if not self.notification_manager:
                logger.warning("Notification manager not available, skipping trade notification")
                return False
                
            # Compose notification message
            message = (
                f"New trade executed:\n"
                f"Strategy: {strategy.name}\n"
                f"Symbol: {signal.symbol}\n"
                f"Side: {signal.side}\n"
                f"Price: {trade_result.price}\n"
                f"Quantity: {trade_result.quantity}\n"
                f"Take Profit: {signal.take_profit}\n"
                f"Stop Loss: {signal.stop_loss}"
            )

            # Send the notification
            self.notification_manager.send_notification(
                user_id=user.id,
                message=message,
                notification_type="trade_execution"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error sending trade notification for user {user.id}: {str(e)}")
            return False
            
    def monitor_positions(self) -> None:
        """
        Monitor open positions and apply risk management rules.
        
        Reviews all active positions for exit conditions, trailing stops,
        and risk exposure adjustments.
        """
        try:
            logger.info("Monitoring open positions")
            
            # Skip if position repository is not available
            if not self.position_repository:
                logger.warning("Position repository not available, skipping position monitoring")
                return
                
            # Get all active positions
            active_positions = self.position_repository.get_all_active_positions()
            if not active_positions:
                logger.info("No active positions to monitor")
                return
                
            # Process each position
            for position in active_positions:
                self._monitor_position(position)
                
            logger.info(f"Monitored {len(active_positions)} positions")
            
        except Exception as e:
            logger.error(f"Error monitoring positions: {str(e)}")
    
    def _monitor_position(self, position) -> None:
        """
        Apply monitoring and risk rules to a specific position.
        
        Args:
            position: Position object to monitor
        """
        try:
            # Skip if user is not initialized
            if position.user_id not in self.user_exchange_map:
                return
                
            # Get exchange for this user
            exchange = self.user_exchange_map[position.user_id]
            
            # Get current market price for position symbol
            current_price = exchange.get_current_price(position.symbol)
            if not current_price:
                return
                
            # Check for take profit or stop loss hits
            if position.side == "buy":  # Long position
                # Check stop loss
                if position.stop_loss and current_price <= position.stop_loss:
                    self._close_position(position, current_price, "stop_loss")
                    return
                    
                # Check take profit
                if position.take_profit and current_price >= position.take_profit:
                    self._close_position(position, current_price, "take_profit")
                    return
                    
                # Check break even
                if hasattr(position, 'break_even_activated') and position.break_even_activated and position.break_even_price and current_price >= position.break_even_price:
                    # Adjust stop loss to entry price
                    self._update_position_stop_loss(position, position.average_entry_price)
                    
                # Check trailing stop
                if hasattr(position, 'trailing_stop') and position.trailing_stop and position.trailing_stop_distance:
                    # Calculate new stop loss based on trailing distance
                    potential_stop = current_price * (1 - position.trailing_stop_distance / 100)
                    if potential_stop > position.stop_loss:
                        # Update to new, higher stop loss
                        self._update_position_stop_loss(position, potential_stop)
                    
            else:  # Short position
                # Check stop loss
                if position.stop_loss and current_price >= position.stop_loss:
                    self._close_position(position, current_price, "stop_loss")
                    return
                    
                # Check take profit
                if position.take_profit and current_price <= position.take_profit:
                    self._close_position(position, current_price, "take_profit")
                    return
                    
                # Check break even
                if hasattr(position, 'break_even_activated') and position.break_even_activated and position.break_even_price and current_price <= position.break_even_price:
                    # Adjust stop loss to entry price
                    self._update_position_stop_loss(position, position.average_entry_price)
                    
                # Check trailing stop
                if hasattr(position, 'trailing_stop') and position.trailing_stop and position.trailing_stop_distance:
                    # Calculate new stop loss based on trailing distance
                    potential_stop = current_price * (1 + position.trailing_stop_distance / 100)
                    if potential_stop < position.stop_loss:
                        # Update to new, lower stop loss
                        self._update_position_stop_loss(position, potential_stop)
                        
        except Exception as e:
            logger.error(f"Error monitoring position {position.id}: {str(e)}")
    
    def _close_position(self, position, current_price, reason) -> bool:
        """
        Close a position at current market price.
        
        Args:
            position: Position to close
            current_price: Current market price
            reason: Reason for closing
            
        Returns:
            bool: Success status
        """
        try:
            # Skip if user is not initialized
            if position.user_id not in self.user_exchange_map:
                logger.warning(f"User {position.user_id} not initialized, cannot close position")
                return False
                
            # Get exchange for this user
            exchange = self.user_exchange_map[position.user_id]
            
            # Execute market close order
            close_result = exchange.execute_market_order(
                symbol=position.symbol,
                side="sell" if position.side == "buy" else "buy",  # Opposite side to close
                quantity=position.current_quantity
            )
            
            if not close_result:
                logger.warning(f"Failed to close position {position.id}")
                return False
                
            # Calculate profit
            entry_price = position.average_entry_price
            close_price = close_result.price
            
            if position.side == "buy":
                profit = (close_price - entry_price) * position.current_quantity
                profit_percentage = (close_price - entry_price) / entry_price * 100
            else:
                profit = (entry_price - close_price) * position.current_quantity
                profit_percentage = (entry_price - close_price) / entry_price * 100
                
            # Update position status in repository
            if self.position_repository:
                self.position_repository.close_position(
                    position_id=position.id,
                    close_price=close_price,
                    profit=profit,
                    profit_percentage=profit_percentage
                )
                
            # Send notification
            if self.notification_manager:
                self.notification_manager.send_notification(
                    user_id=position.user_id,
                    message=(
                        f"Position closed: {position.symbol}\n"
                        f"Reason: {reason}\n"
                        f"Entry: {entry_price}\n"
                        f"Exit: {close_price}\n"
                        f"P/L: {profit_percentage:.2f}%"
                    ),
                    notification_type=f"{reason}_hit"
                )
                
            logger.info(f"Closed position {position.id} at {close_price}, P/L: {profit_percentage:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {position.id}: {str(e)}")
            return False
    
    def _update_position_stop_loss(self, position, new_stop_loss) -> bool:
        """
        Update stop loss level for a position.
        
        Args:
            position: Position to update
            new_stop_loss: New stop loss price
            
        Returns:
            bool: Success status
        """
        try:
            # Update in database
            if self.position_repository:
                result = self.position_repository.update_stop_loss(position.id, new_stop_loss)
                if result:
                    logger.info(f"Updated stop loss for position {position.id} to {new_stop_loss}")
                    return True
                    
            logger.warning(f"Failed to update stop loss for position {position.id}")
            return False
            
        except Exception as e:
            logger.error(f"Error updating stop loss for position {position.id}: {str(e)}")
            return False
            
    def check_risk_exposure(self) -> Dict[str, Any]:
        """
        Check current risk exposure across all users and positions.
        
        Analyzes portfolio risk levels, concentration risks, and overall
        exposure metrics to ensure compliance with risk parameters.
        
        Returns:
            Dict with risk exposure metrics
        """
        try:
            logger.info("Checking risk exposure")
            
            # Skip if position repository is not available
            if not self.position_repository:
                logger.warning("Position repository not available, skipping risk exposure check")
                return {}
                
            # Get all active positions
            active_positions = self.position_repository.get_all_active_positions()
            if not active_positions:
                logger.info("No active positions for risk assessment")
                return {"total_exposure": 0, "max_drawdown_risk": 0, "users_at_risk": 0}
                
            # Calculate risk metrics
            total_exposure = sum(p.current_quantity * p.average_entry_price for p in active_positions)
            
            # Identify positions at high risk
            high_risk_positions = self.position_repository.get_positions_at_risk(threshold_percentage=3.0)
            
            # Group positions by user
            user_positions = {}
            for position in active_positions:
                if position.user_id not in user_positions:
                    user_positions[position.user_id] = []
                user_positions[position.user_id].append(position)
                
            # Count users at or near max risk
            users_at_risk = 0
            for user_id, positions in user_positions.items():
                user_exposure = sum(p.current_quantity * p.average_entry_price for p in positions)
                # Determine if user is at risk (implementation details depend on user risk profile)
                # This is a simplified check
                if user_exposure > 5000:  # Example threshold
                    users_at_risk += 1
                    
            # Simulate potential max drawdown
            max_drawdown_risk = self._calculate_max_drawdown_risk(active_positions)
            
            risk_report = {
                "total_exposure": total_exposure,
                "positions_count": len(active_positions),
                "high_risk_positions": len(high_risk_positions),
                "users_at_risk": users_at_risk,
                "max_drawdown_risk": max_drawdown_risk
            }
            
            logger.info(f"Risk assessment completed: {risk_report}")
            return risk_report
            
        except Exception as e:
            logger.error(f"Error checking risk exposure: {str(e)}")
            return {}
    
    def _calculate_max_drawdown_risk(self, positions) -> float:
        """
        Calculate potential maximum drawdown risk.
        
        Args:
            positions: List of active positions
            
        Returns:
            float: Estimated maximum drawdown as percentage
        """
        try:
            if not positions:
                return 0.0
                
            # Simple simulation - assumes worst case where all positions hit stop loss
            total_value = sum(p.current_quantity * p.average_entry_price for p in positions)
            if total_value == 0:
                return 0.0
                
            potential_loss = 0
            for position in positions:
                if position.stop_loss:
                    if position.side == "buy":
                        position_loss = (position.average_entry_price - position.stop_loss) * position.current_quantity
                    else:
                        position_loss = (position.stop_loss - position.average_entry_price) * position.current_quantity
                    potential_loss += max(0, position_loss)
                else:
                    # If no stop loss, assume 100% loss (worst case)
                    potential_loss += position.current_quantity * position.average_entry_price
                    
            return (potential_loss / total_value) * 100
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown risk: {str(e)}")
            return 0.0
    
    def reset_engine(self) -> bool:
        """
        Perform a soft reset of the trading engine.
        
        Clears caches and reconnects to exchanges without stopping the engine.
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("Performing trading engine reset")
            
            with self.lock:
                # Clear caches
                self.market_data_cache.clear()
                self.active_users_cache.clear()
                self.error_counts.clear()
                
                # Reset timestamps
                self.last_market_update = 0
                self.last_user_refresh = 0
                
                # Reset schema verification flag
                self.schema_verified = False
                
                # Verify database schema again
                self._verify_database_schema()
                
                # Reestablish all exchange connections
                for user_id in list(self.user_exchange_map.keys()):
                    try:
                        # Get exchange details
                        old_exchange = self.user_exchange_map[user_id]
                        exchange_name = old_exchange.name
                        
                        # Get API keys from manager
                        if self.api_key_manager:
                            api_keys = self.api_key_manager.get_api_keys(user_id)
                            if api_keys and self.exchange_factory:
                                # Close old connection
                                try:
                                    old_exchange.close()
                                except:
                                    pass
                                    
                                # Create new connection
                                new_exchange = self.exchange_factory.create_exchange(
                                    exchange_name,
                                    api_keys['api_key'],
                                    api_keys['secret_key']
                                )
                                
                                # Initialize and replace
                                if new_exchange.initialize():
                                    self.user_exchange_map[user_id] = new_exchange
                                    logger.info(f"Reconnected exchange for user {user_id}")
                    except Exception as e:
                        logger.error(f"Error reconnecting exchange for user {user_id}: {str(e)}")
                
                # Reset health status
                if self.running:
                    self.health_status = "running"
                
                logger.info("Trading engine reset completed")
                return True
                
        except Exception as e:
            logger.error(f"Error resetting trading engine: {str(e)}")
            self.health_status = "degraded"
            return False

# Maintain backward compatibility
Engine = TradingEngine