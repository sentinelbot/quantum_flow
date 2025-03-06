# core/app.py
import logging
import signal
import sys
import time
from threading import Event
from typing import Optional
import datetime

from config.app_config import AppConfig
from core.engine import TradingEngine
from core.scheduler import Scheduler
from database.db import DatabaseManager
from database.repository.user_repository import UserRepository
from exchange.exchange_factory import ExchangeFactory
from notification.notification_manager import NotificationManager
from security.api_key_manager import APIKeyManager
from maintenance.self_healing import SelfHealingSystem
from maintenance.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)

class QuantumFlow:
    """
    Main application class for QuantumFlow Trading Bot
    """
    def __init__(self, 
                trading_engine: TradingEngine,
                notification_manager: NotificationManager,
                scheduler: Scheduler,
                user_repository: UserRepository,
                config: AppConfig):
        """
        Initialize the QuantumFlow application with dependencies.
        
        Args:
            trading_engine: Trading engine for executing trades
            notification_manager: System for handling notifications
            scheduler: Task scheduler
            user_repository: Repository for user management
            config: Application configuration
        """
        self.config = config
        self.trading_engine = trading_engine
        self.notification_manager = notification_manager
        self.scheduler = scheduler
        self.user_repository = user_repository
        self.shutdown_event = Event()
        self.start_time = None
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        logger.info("QuantumFlow application initialized")
        
    def start(self):
        """
        Start the QuantumFlow trading bot
        """
        logger.info("Starting QuantumFlow Trading Bot")
        self.start_time = datetime.datetime.now()
        
        # Start notification system
        self.notification_manager.start()
        
        # Start trading engine
        self.trading_engine.start()
        
        # Start scheduler
        self.scheduler.start()
        
        logger.info("QuantumFlow Trading Bot started successfully")
        
        # Notify admins about successful startup
        self.notify_startup()
        
        # Keep main thread alive until shutdown
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.handle_shutdown(None, None)
            
    def handle_shutdown(self, sig, frame):
        """
        Handle graceful shutdown of the application
        """
        if self.shutdown_event.is_set():
            logger.warning("Forced shutdown initiated")
            sys.exit(1)
            
        logger.info("Graceful shutdown initiated")
        self.shutdown_event.set()
        
        # Stop all components in reverse order
        self.scheduler.stop()
        self.trading_engine.stop()
        self.notification_manager.stop()
        
        logger.info("QuantumFlow Trading Bot shutdown complete")
    
    def get_uptime(self) -> int:
        """
        Get the application uptime in seconds
        
        Returns:
            int: Uptime in seconds, or 0 if not started
        """
        if not self.start_time:
            return 0
            
        delta = datetime.datetime.now() - self.start_time
        return int(delta.total_seconds())
        
    def notify_startup(self):
        """
        Send startup notification to administrators
        """
        try:
            admin_users = self.user_repository.get_admin_users()
            
            startup_message = (
                f"QuantumFlow Trading Bot started successfully at "
                f"{self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            for admin in admin_users:
                self.notification_manager.send_notification(
                    user_id=admin.id,
                    message=startup_message,
                    notification_type="system_status"
                )
                
        except Exception as e:
            logger.error(f"Failed to send startup notification: {str(e)}")