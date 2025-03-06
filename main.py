"""
QuantumFlow Trading Bot - Main Application

This is the central integration point for the QuantumFlow trading bot system,
coordinating all components and providing the primary entry point for the application.
"""

import logging
import time
import signal
import threading
import traceback
from typing import Dict, List, Any, Optional

# Core components
from core.app import QuantumFlow as Application
from core.engine import TradingEngine
from core.scheduler import Scheduler as TaskScheduler

# Configuration
from config.app_config import AppConfig
from config.logging_config import configure_logging
from config.trading_config import TradingConfig

# Database
from database.db import DatabaseManager
from database.repository.user_repository import UserRepository
from database.repository.trade_repository import TradeRepository
from database.repository.position_repository import PositionRepository
from database.repository.analytics_repository import AnalyticsRepository
from database.repository.api_key_repository import ApiKeyRepository
from database.models.api_key import ApiKey

# Exchange
from exchange.exchange_factory import ExchangeFactory
from exchange.exchange_helper import ExchangeHelper

# Strategies
from strategies.strategy_factory import StrategyFactory

# Risk management
from risk.risk_manager import RiskManager
from risk.position_sizer import PositionSizer
from risk.drawdown_protector import DrawdownProtector

# Analysis
from analysis.market_analyzer import MarketAnalyzer
from analysis.technical_indicators import TechnicalIndicators
from analysis.sentiment_analyzer import SentimentAnalyzer

# Machine Learning
from ml.data_processor import DataProcessor
from ml.model_trainer import ModelTrainer
from ml.models.price_predictor import PricePredictor
from ml.models.pattern_recognition import PatternRecognition
from ml.models.regime_classifier import RegimeClassifier

# Notification
from notification.telegram_bot import TelegramBot
from notification.email_notifier import EmailNotifier
from notification.notification_manager import NotificationManager

# Admin
from admin.admin_bot import AdminBot
from admin.dashboard_api import DashboardAPI
from admin.monitoring import SystemMonitoring

# Security
from security.encryption import EncryptionService
from security.api_key_manager import APIKeyManager
from security.auth import AuthenticationService

# Compliance
from compliance.kyc import KYCProcessor
from compliance.aml import AMLChecker
from compliance.reporting import ComplianceReporter

# Profit
from profit.fee_calculator import FeeCalculator
from profit.profit_tracker import ProfitTracker

# Maintenance
from maintenance.self_healing import SelfHealingSystem
from maintenance.system_monitor import SystemMonitor

# Utils
from utils.helpers import generate_unique_id
from utils.validators import validate_config
from utils.decorators import log_execution, retry


class QuantumFlowBot:
   """
   Main QuantumFlow Trading Bot class that integrates all system components
   and manages their lifecycle.
   """
   
   def __init__(self, config_path: str = "config/settings.json"):
       """
       Initialize the QuantumFlow Trading Bot with all its components.
       
       Args:
           config_path: Path to the configuration file
       """
       # Configure logging first
       configure_logging()
       self.logger = logging.getLogger(__name__)
       self.logger.info("Initializing QuantumFlow Trading Bot...")
       
       # Load configuration
       self.app_config = AppConfig(config_path)
       self.trading_config = TradingConfig(config_path)
       
       # Validate configuration
       validation_result = validate_config(self.app_config.get_all())
       if isinstance(validation_result, tuple):
           is_valid, error_message = validation_result
           if not is_valid:
               raise ValueError(f"Invalid configuration detected: {error_message}")
       elif not validation_result:
           raise ValueError("Invalid configuration detected")
       
       # System state
       self.running = False
       self.initialized = False
       self.threads = []
       self.components = {}
       self.system_status = "initializing"
       
       # Setup signal handlers for graceful shutdown
       signal.signal(signal.SIGINT, self.handle_shutdown_signal)
       signal.signal(signal.SIGTERM, self.handle_shutdown_signal)
       
       # Initialize but don't start components yet
       self._init_components()
   
   @log_execution
   def _init_components(self):
       """Initialize all system components without starting them."""
       try:
           self.logger.info("Initializing system components...")
           
           # Initialize core database
           try:
               db_url = f"postgresql://{self.app_config.get('database.username')}:{self.app_config.get('database.password')}@{self.app_config.get('database.host')}:{self.app_config.get('database.port')}/{self.app_config.get('database.name')}"
               self.db_manager = DatabaseManager(db_url)
               # Make sure database is properly initialized and schema is validated
               if not self.db_manager.initialize():
                   raise RuntimeError("Database initialization failed")
               
               # Verify the schema immediately after initialization
               # This will add missing columns and avoid errors later
               self._verify_database_schema()
               
           except Exception as e:
               self.logger.error(f"Failed to initialize database: {str(e)}")
               raise RuntimeError("Database initialization failed") from e
           
           # Initialize repositories - use enhanced versions with schema adaptation
           self.user_repo = UserRepository(self.db_manager)
           self.trade_repo = TradeRepository(self.db_manager)
           self.position_repo = PositionRepository(self.db_manager)
           self.analytics_repo = AnalyticsRepository(self.db_manager)
           
           # Initialize security services
           jwt_secret = self.app_config.get("security.jwt_secret")
           if not jwt_secret:
               raise ValueError("JWT secret is required but not found in configuration")
               
           self.encryption_service = EncryptionService(
               key=self.app_config.get("security.encryption_key")
           )
           self.auth_service = AuthenticationService(
               db=self.db_manager,
               jwt_secret=self.app_config.get("security.jwt_secret")
           )
           self.api_key_manager = APIKeyManager(
               db=self.db_manager,
           )
           
           # Initialize API key repository
           self.api_key_repository = ApiKeyRepository(
               db_manager=self.db_manager,
               encryption_service=self.encryption_service
           )
           
           # Initialize exchange components
           self.exchange_factory = ExchangeFactory()
           # Get a default exchange instance for analysis
           default_exchange = self.exchange_factory.create_exchange(
               exchange_name=self.app_config.get("exchange.default", "binance"),
               api_key=self.app_config.get("exchange.api_key", ""),
               api_secret=self.app_config.get("exchange.api_secret", "")
           )
           
           # Initialize notification components first (since other components depend on them)
           self.email_notifier = EmailNotifier(
               enabled=self.app_config.get("notification.email.enabled", False),
               smtp_server=self.app_config.get("notification.email.smtp_server", ""),
               smtp_port=self.app_config.get("notification.email.smtp_port", 587),
               smtp_username=self.app_config.get("notification.email.smtp_username", ""),
               smtp_password=self.app_config.get("notification.email.smtp_password", ""),
               from_email=self.app_config.get("notification.email.from_email", "")
           )
           self.notification_manager = NotificationManager(
               config=self.app_config.get("notification.general")
           )
           
           # Initialize risk management components
           self.position_sizer = PositionSizer(config=self.app_config)
           self.drawdown_protector = DrawdownProtector(
               config=self.app_config,
               db=self.db_manager,
               notification_manager=self.notification_manager
           )
           self.risk_manager = RiskManager(
               config=self.app_config,
               db=self.db_manager
           )
           
           # Initialize strategy factory with required parameters
           self.strategy_factory = StrategyFactory(
               config=self.app_config,  # Pass AppConfig instance
               risk_manager=self.risk_manager  # Pass RiskManager instance
           )
           
           # Initialize analysis tools
           self.technical_indicators = TechnicalIndicators()
           self.sentiment_analyzer = SentimentAnalyzer()
           # Market analyzer requires an exchange instance
           self.market_analyzer = MarketAnalyzer(
               exchange=default_exchange
           )
           
           # Initialize ML components
           # DataProcessor initialization
           data_processing_config = self.app_config.get("ml.data_processing", {})
           cache_dir = data_processing_config.get("cache_dir", "cache") if isinstance(data_processing_config, dict) else "cache"
           self.data_processor = DataProcessor(
               cache_dir=cache_dir
           )
           
           # PricePredictor initialization
           price_predictor_config = self.app_config.get("ml.models.price_predictor", {})
           price_model_dir = price_predictor_config.get("model_dir", "models") if isinstance(price_predictor_config, dict) else "models"
           self.price_predictor = PricePredictor(
               model_dir=price_model_dir
           )
           
           # PatternRecognition initialization
           pattern_recognition_config = self.app_config.get("ml.models.pattern_recognition", {})
           pattern_model_dir = pattern_recognition_config.get("model_dir", "models") if isinstance(pattern_recognition_config, dict) else "models"
           self.pattern_recognition = PatternRecognition(
               model_dir=pattern_model_dir
           )
           
           # RegimeClassifier initialization
           regime_classifier_config = self.app_config.get("ml.models.regime_classifier", {})
           regime_model_dir = regime_classifier_config.get("model_dir", "models") if isinstance(regime_classifier_config, dict) else "models"
           self.regime_classifier = RegimeClassifier(
               model_dir=regime_model_dir
           )
           # Initialize ML model trainer
           training_config = self.app_config.get("ml.training", {})
           model_dir = training_config.get("model_dir", "models") if isinstance(training_config, dict) else "models"
           data_dir = training_config.get("data_dir", "data") if isinstance(training_config, dict) else "data"
           
           self.model_trainer = ModelTrainer(
               model_dir=model_dir,
               data_dir=data_dir
           )
           
           # Initialize telegram bot 
           telegram_token = self.app_config.get("notification.telegram.token", "")
           admin_chat_ids = self.app_config.get("admin.admin_user_ids", [])
           
           self.telegram_bot = TelegramBot(
               token=telegram_token,
               admin_chat_ids=admin_chat_ids,
               api_key_repository=self.api_key_repository
           )
           
           # Initialize Exchange Helper
           self.exchange_helper = ExchangeHelper(
               exchange_factory=self.exchange_factory,
               api_key_repository=self.api_key_repository,
               user_repository=self.user_repo
           )
           
           # Initialize admin components
           self.system_monitoring = SystemMonitoring(
               notification_manager=self.notification_manager
           )
           self.admin_bot = AdminBot(
               token=self.app_config.get("admin.telegram.token"),
               admin_ids=self.app_config.get("admin.admin_user_ids"),
               user_repository=self.user_repo,
               trade_repository=self.trade_repo,
               system_monitoring=self.system_monitoring
           )
           self.dashboard_api = DashboardAPI(
               auth_service=self.auth_service,
               user_repository=self.user_repo,
               trade_repository=self.trade_repo,
               analytics_repository=self.analytics_repo,
               system_monitoring=self.system_monitoring,
               host=self.app_config.get("admin.dashboard.host"),
               port=self.app_config.get("admin.dashboard.port")
           )
           
           # Initialize compliance components
           self.kyc_processor = KYCProcessor(
               user_repository=self.user_repo,
               notification_manager=self.notification_manager,
               config=self.app_config.get("compliance.kyc")
           )
           self.aml_checker = AMLChecker(
               user_repository=self.user_repo,
               trade_repository=self.trade_repo,
               notification_manager=self.notification_manager,
               config=self.app_config.get("compliance.aml")
           )
           self.compliance_reporter = ComplianceReporter(
               user_repository=self.user_repo,
               trade_repository=self.trade_repo,
               config=self.app_config.get("compliance.reporting")
           )
           
           # Initialize profit tracking
           self.fee_calculator = FeeCalculator(
               config=self.app_config.get("profit.fees")
           )
           self.profit_tracker = ProfitTracker(
               fee_calculator=self.fee_calculator,
               trade_repository=self.trade_repo,
               user_repository=self.user_repo,
               config=self.app_config.get("profit.tracking")
           )
           
           # Initialize system monitor first (needed for self-healing)
           self.system_monitor = SystemMonitor(
               notification_manager=self.notification_manager
           )
           
           # Initialize scheduler (needed for many components)
           self.scheduler = TaskScheduler()
           
           # Initialize trading engine (depends on many components)
           self.trading_engine = TradingEngine(
               config=self.app_config,
               trading_config=self.trading_config,
               db=self.db_manager,
               exchange_factory=self.exchange_factory,
               strategy_factory=self.strategy_factory,
               api_key_manager=self.api_key_manager,
               risk_manager=self.risk_manager,
               notification_manager=self.notification_manager,
               market_analyzer=self.market_analyzer,
               price_predictor=self.price_predictor,
               pattern_recognition=self.pattern_recognition,
               regime_classifier=self.regime_classifier,
               position_repository=self.position_repo,
               trade_repository=self.trade_repo,
               user_repository=self.user_repo,
               profit_tracker=self.profit_tracker
           )
           
           # Initialize self-healing AFTER trading engine
           # This ensures it has a valid reference
           self.self_healing = SelfHealingSystem(
               config=self.app_config,
               trading_engine=self.trading_engine,
               system_monitor=self.system_monitor,
               notification_manager=self.notification_manager
           )
           
           # Initialize core application (coordinates all components)
           self.app = Application(
               trading_engine=self.trading_engine,
               notification_manager=self.notification_manager,
               scheduler=self.scheduler,
               user_repository=self.user_repo,
               config=self.app_config
           )
           
           # Track components for status reporting and management
           self.components = {
               "database": self.db_manager,
               "trading_engine": self.trading_engine,
               "telegram_bot": self.telegram_bot,
               "admin_bot": self.admin_bot,
               "dashboard_api": self.dashboard_api,
               "notification_manager": self.notification_manager,
               "system_monitor": self.system_monitor,
               "self_healing": self.self_healing,
               "scheduler": self.scheduler,
               "app": self.app
           }
           
           self.initialized = True
           self.logger.info("All components initialized successfully")
           
       except Exception as e:
           self.logger.critical(f"Failed to initialize components: {str(e)}", exc_info=True)
           self.system_status = "initialization_error"
           raise

   def _verify_database_schema(self):
       """Verify and fix database schema issues.
       """
       try:
           self.logger.info("Verifying database schema...")
           
           # Check if the database has schema verification capability
           if hasattr(self.db_manager, '_validate_and_fix_schema'):
               self.db_manager._validate_and_fix_schema()
               self.logger.info("Database schema verified and fixed if needed")
           else:
               # Manual schema verification for critical tables
               conn = self.db_manager.get_connection()
               cursor = conn.cursor()
               
               # Check users table for missing columns
               try:
                   # Check if is_admin column exists
                   cursor.execute("""
                       SELECT column_name 
                       FROM information_schema.columns 
                       WHERE table_name = 'users' AND column_name = 'is_admin';
                   """)
                   
                   if cursor.fetchone() is None:
                       self.logger.warning("Missing 'is_admin' column in users table, adding it now")
                       cursor.execute("""
                           ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT false;
                       """)
                       
                   # Check if is_active column exists
                   cursor.execute("""
                       SELECT column_name 
                       FROM information_schema.columns 
                       WHERE table_name = 'users' AND column_name = 'is_active';
                   """)
                   
                   if cursor.fetchone() is None:
                       self.logger.warning("Missing 'is_active' column in users table, adding it now")
                       cursor.execute("""
                           ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true;
                       """)
                   
                   # Check if is_paused column exists
                   cursor.execute("""
                       SELECT column_name 
                       FROM information_schema.columns 
                       WHERE table_name = 'users' AND column_name = 'is_paused';
                   """)
                   
                   if cursor.fetchone() is None:
                       self.logger.warning("Missing 'is_paused' column in users table, adding it now")
                       cursor.execute("""
                           ALTER TABLE users ADD COLUMN IF NOT EXISTS is_paused BOOLEAN DEFAULT false;
                       """)
                   
                   # Check api_keys table existence
                   cursor.execute("""
                       SELECT EXISTS (
                           SELECT FROM information_schema.tables 
                           WHERE table_name = 'api_keys'
                       );
                   """)
                   
                   if not cursor.fetchone()[0]:
                       self.logger.warning("api_keys table doesn't exist, creating it now")
                       cursor.execute("""
                           CREATE TABLE api_keys (
                               id VARCHAR(36) PRIMARY KEY,
                               user_id INTEGER NOT NULL REFERENCES users(id),
                               exchange VARCHAR(50) NOT NULL,
                               encrypted_api_key TEXT NOT NULL,
                               encrypted_api_secret TEXT NOT NULL,
                               is_active BOOLEAN DEFAULT TRUE,
                               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                               updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                           );
                       """)
                       
                   conn.commit()
                   self.logger.info("Database schema verification completed")
                   
               except Exception as e:
                   conn.rollback()
                   self.logger.error(f"Error verifying database schema: {str(e)}")
               finally:
                   cursor.close()
                   self.db_manager.release_connection(conn)
                   
       except Exception as e:
           self.logger.error(f"Database schema verification failed: {str(e)}")
   
   def _register_scheduled_tasks(self):
       """Register all scheduled tasks with the scheduler."""
       self.logger.info("Registering scheduled tasks...")
       
       # Trading engine maintenance tasks
       self.scheduler.schedule(
           func=self.trading_engine.update_market_data,
           interval_seconds=300,  # 5 minutes
           task_id="update_market_data"
       )
       
       # ML model updates
       self.scheduler.schedule(
           func=self.model_trainer.update_models,
           interval_seconds=86400,  # Daily
           task_id="update_ml_models"
       )
       
       # Risk assessment
       self.scheduler.schedule(
           func=self.risk_manager.assess_portfolio_risk,
           interval_seconds=3600,  # Hourly
           task_id="assess_portfolio_risk"
       )
       
       # Compliance checks
       self.scheduler.schedule(
           func=self.aml_checker.run_periodic_checks,
           interval_seconds=14400,  # 4 hours
           task_id="aml_checks"
       )
       
       # Performance reporting
       self.scheduler.schedule(
           func=self.profit_tracker.generate_daily_report,
           interval_seconds=86400,  # Daily
           task_id="daily_profit_report",
           run_at_time="00:05:00"  # Run at 5 minutes past midnight
       )
       
       # System maintenance
       self.scheduler.schedule(
           func=self.db_manager.optimize_database,
           interval_seconds=86400,  # Daily
           task_id="db_optimization",
           run_at_time="03:00:00"  # Run at 3 AM
       )
       
       self.logger.info("All scheduled tasks registered")
   
   @retry(max_attempts=3, delay=2.0, backoff=2.0)
   def _start_component(self, name: str, component: Any, start_method: str = "start"):
       """
       Start a single component with retry logic.
       
       Args:
           name: Component name for logging
           component: Component object
           start_method: Name of the method to call to start the component
       """
       try:
           self.logger.info(f"Starting component: {name}")
           start_func = getattr(component, start_method)
           start_func()
           self.logger.info(f"Component {name} started successfully")
           return True
       except Exception as e:
           self.logger.error(f"Failed to start component {name}: {str(e)}", exc_info=True)
           raise
   
   def _start_threaded_component(self, name: str, component: Any, start_method: str = "start"):
       """
       Start a component in a separate thread.
       
       Args:
           name: Component name for logging
           component: Component object
           start_method: Name of the method to call to start the component
       """
       def thread_target():
           try:
               start_func = getattr(component, start_method)
               start_func()
           except Exception as e:
               self.logger.error(f"Error in threaded component {name}: {str(e)}", exc_info=True)
               # Notify admin about component failure
               self.notification_manager.send_admin_alert(
                   f"Component Failure: {name}",
                   f"Component {name} has failed with error: {str(e)}. System trying to recover."
               )
               # Request self-healing
               self.self_healing.heal_component(name, component)
       
       thread = threading.Thread(target=thread_target, name=f"{name}_thread", daemon=True)
       thread.start()
       self.threads.append((name, thread))
       self.logger.info(f"Component {name} started in separate thread")
       return thread
   
   @log_execution
   def start(self):
       """Start the QuantumFlow Trading Bot system."""
       if not self.initialized:
           self.logger.error("Cannot start: system not fully initialized")
           raise RuntimeError("System not initialized")
       
       if self.running:
           self.logger.warning("System is already running")
           return
       
       self.logger.info("Starting QuantumFlow Trading Bot...")
       
       try:
           # Start order is important - dependencies first
           
           # 1. Database connection
           self._start_component("database", self.db_manager, "connect")
           
           # 2. System monitor first (for monitoring other components)
           self._start_threaded_component("system_monitor", self.system_monitor)
           
           # 3. Core services 
           self._start_component("scheduler", self.scheduler)
           self._register_scheduled_tasks()
           
           # 4. Dashboard API (background service)
           self._start_threaded_component("dashboard_api", self.dashboard_api, "start_server")
           
           # 5. Notification systems
           self._start_component("notification_manager", self.notification_manager)
           self._start_threaded_component("telegram_bot", self.telegram_bot)
           self._start_threaded_component("admin_bot", self.admin_bot)
           
           # 6. Trading engine (core functionality)
           # Important: Start this before self_healing so engine is running when self_healing accesses it
           self._start_component("trading_engine", self.trading_engine)
           
           # 7. Self-healing system (after trading engine is running)
           self._start_component("self_healing", self.self_healing)
           
           # 8. Update self-healing with running trading engine
           self.self_healing.update_component("trading_engine", self.trading_engine)
           
           # 9. Finally start the application coordinator
           self._start_component("app", self.app)
           
           self.running = True
           self.system_status = "running"
           
           self.logger.info("QuantumFlow Trading Bot successfully started!")
           
           # Send startup notification to admins
           self.notification_manager.send_admin_alert(
               "System Started",
               f"QuantumFlow Trading Bot has been successfully started at {time.strftime('%Y-%m-%d %H:%M:%S')}"
           )
           
           return True
           
       except Exception as e:
           self.system_status = "error"
           self.logger.critical(f"Failed to start system: {str(e)}", exc_info=True)
           # Try to notify admins about startup failure
           try:
               self.notification_manager.send_admin_alert(
                   "CRITICAL: System Startup Failed",
                   f"QuantumFlow Trading Bot failed to start: {str(e)}"
               )
           except:
               pass
           
           # Try to gracefully shut down any started components
           self.stop()
           raise
   
   @log_execution
   def stop(self):
       """Stop the QuantumFlow Trading Bot system gracefully."""
       if not self.running:
           self.logger.info("System is not running, nothing to stop")
           return
       
       self.logger.info("Stopping QuantumFlow Trading Bot...")
       self.system_status = "stopping"
       
       # Stop components in reverse order (dependencies last)
       stop_order = [
           "app",
           "self_healing",
           "trading_engine", 
           "admin_bot",
           "telegram_bot", 
           "notification_manager",
           "dashboard_api",
           "scheduler",
           "system_monitor",
           "database"
       ]
       
       # Track failed components
       failed_components = []
       
       # First, stop each component
       for component_name in stop_order:
           if component_name in self.components:
               component = self.components[component_name]
               try:
                   self.logger.info(f"Stopping component: {component_name}")
                   if hasattr(component, "stop"):
                       component.stop()
                   self.logger.info(f"Component {component_name} stopped successfully")
               except Exception as e:
                   self.logger.error(f"Error stopping component {component_name}: {str(e)}", exc_info=True)
                   failed_components.append(component_name)
       
       # Wait for threaded components to terminate
       for thread_name, thread in self.threads:
           try:
               self.logger.info(f"Waiting for thread {thread_name} to terminate...")
               thread.join(timeout=10.0)  # Give threads 10 seconds to terminate
               if thread.is_alive():
                   self.logger.warning(f"Thread {thread_name} did not terminate in time")
           except Exception as e:
               self.logger.error(f"Error waiting for thread {thread_name}: {str(e)}")
       
       self.threads = []
       self.running = False
       
       if failed_components:
           self.system_status = "error"
           error_msg = f"System stopped with errors in components: {', '.join(failed_components)}"
           self.logger.warning(error_msg)
           return False
       else:
           self.system_status = "stopped"
           self.logger.info("QuantumFlow Trading Bot successfully stopped")
           return True
   
   def handle_shutdown_signal(self, signum, frame):
       """Handle shutdown signals from the OS."""
       signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
       self.logger.info(f"Received {signal_name} signal, initiating graceful shutdown...")
       self.stop()
   
   def get_system_status(self) -> Dict[str, Any]:
       """Get the current system status information."""
       status = {
           "status": self.system_status,
           "initialized": self.initialized,
           "running": self.running,
           "start_time": self.app.start_time if hasattr(self.app, "start_time") else None,
           "uptime_seconds": self.app.get_uptime() if hasattr(self.app, "get_uptime") else 0,
           "components": {}
       }
       
       # Collect status from individual components where available
       for name, component in self.components.items():
           if hasattr(component, "get_status"):
               status["components"][name] = component.get_status()
           else:
               status["components"][name] = "unknown"
               
       return status
   
   def run_forever(self):
       """Run the bot forever until interrupted."""
       if not self.running:
           self.start()
       
       self.logger.info("QuantumFlow Trading Bot is running. Press Ctrl+C to stop.")
       
       # Main loop
       try:
           while self.running:
               time.sleep(1)
       except KeyboardInterrupt:
           self.logger.info("Keyboard interrupt received")
       finally:
           self.stop()


# Entry point for the application
if __name__ == "__main__":
   try:
       bot = QuantumFlowBot("config/settings.json")  # Explicitly specify the config path
       bot.run_forever()
   except FileNotFoundError as e:
       logging.critical(f"Configuration file not found: {str(e)}")
       exit(1)
   except ValueError as e:
       logging.critical(f"Invalid configuration: {str(e)}")
       exit(1)
   except Exception as e:
       logging.critical(f"Unhandled exception: {str(e)}", exc_info=True)
       exit(1)