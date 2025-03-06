"""
Advanced Self-Healing System for Automated System Recovery and Resilience Management

This module provides a sophisticated self-healing mechanism to monitor, 
diagnose, and automatically recover critical system components.
"""

import logging
import threading
import time
import inspect
from typing import Dict, Any, Optional, Callable, List, Tuple

logger = logging.getLogger(__name__)

class SelfHealingSystem:
    """
    Comprehensive Self-Healing and System Recovery Management System

    Provides an advanced, configurable approach to system monitoring, 
    fault detection, and automated recovery strategies.
    """

    def __init__(
        self, 
        config=None, 
        trading_engine=None, 
        system_monitor=None,
        notification_manager=None
    ):
        """
        Initialize the Self-Healing System with configurable dependencies.

        Args:
            config (dict): System configuration settings
            trading_engine (TradingEngine): Core trading engine component
            system_monitor (SystemMonitor): System performance monitoring component
            notification_manager (NotificationManager): Notification dispatch system
        """
        # Core system dependencies
        self.config = config or {}
        self.trading_engine = trading_engine
        self.system_monitor = system_monitor
        self.notification_manager = notification_manager

        # Self-healing configuration
        self.healing_interval = self.config.get('self_healing.interval', 10)
        self.max_recovery_attempts = self.config.get('self_healing.max_attempts', 5)

        # Monitoring state
        self.healing_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Component recovery tracking
        self.recovery_history = {}
        self.component_states = {}
        
        # Component existence flags
        self.has_trading_engine = self.trading_engine is not None
        self.has_system_monitor = self.system_monitor is not None
        self.has_notification_manager = self.notification_manager is not None

        # Initialize healing strategies
        self.healing_strategies = self._configure_healing_strategies()

        logger.info("Self-Healing System initialized successfully")

    def _configure_healing_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Configure comprehensive healing strategies for critical system components.

        Returns:
            Dict of configurable healing strategies with check and recovery methods
        """
        strategies = {
            "database_connection": {
                "check": self._check_database_connection,
                "heal": self._heal_database_connection,
                "severity": "critical",
                "attempts": 0,
                "max_attempts": 5,
                "backoff_factor": 2.0,
                "initial_delay": 5.0
            },
            "api_connections": {
                "check": self._check_api_connections,
                "heal": self._heal_api_connections,
                "severity": "critical",
                "attempts": 0,
                "max_attempts": 5,
                "backoff_factor": 2.0,
                "initial_delay": 5.0
            }
        }
        
        # Only add trading_engine strategy if the engine exists
        if self.has_trading_engine:
            strategies["trading_engine"] = {
                "check": self._check_trading_engine,
                "heal": self._heal_trading_engine,
                "severity": "critical",
                "attempts": 0,
                "max_attempts": 3,
                "backoff_factor": 2.0,
                "initial_delay": 10.0
            }
            
        return strategies

    def start(self) -> None:
        """
        Activate the self-healing monitoring process.

        Initiates a background thread to continuously monitor 
        and recover system components.
        """
        if self.running:
            logger.warning("Self-Healing System is already operational")
            return
            
        logger.info("Initiating Self-Healing System")
        self.running = True
        self.healing_thread = threading.Thread(target=self._healing_loop, name="self_healing_thread")
        self.healing_thread.daemon = True
        self.healing_thread.start()
        
        logger.info("Self-Healing System activated successfully")
    
    def stop(self) -> None:
        """
        Terminate the self-healing monitoring process.

        Gracefully stops the monitoring thread and releases resources.
        """
        if not self.running:
            logger.warning("Self-Healing System is already inactive")
            return
            
        logger.info("Stopping Self-Healing System")
        self.running = False
        
        if self.healing_thread:
            self.healing_thread.join(timeout=5)
            
        logger.info("Self-Healing System deactivated successfully")
    
    def _healing_loop(self) -> None:
        """
        Continuous monitoring and recovery loop.

        Systematically checks system components and initiates 
        recovery processes when necessary.
        """
        while self.running:
            try:
                self._run_healing_cycle()
                time.sleep(self.healing_interval)
            except Exception as e:
                logger.error(f"Critical error in healing cycle: {str(e)}")
                time.sleep(30)  # Longer sleep on error to prevent rapid failure loops
    
    def _run_healing_cycle(self) -> None:
        """
        Execute a comprehensive system health check and recovery cycle.

        Iterates through defined healing strategies, checking and 
        attempting to recover critical system components.
        """
        for name, strategy in self.healing_strategies.items():
            try:
                # Skip checking components that don't exist or are optional
                if name == "trading_engine" and not self.has_trading_engine:
                    continue
                    
                # Perform health check
                is_healthy = strategy["check"]()
                
                if is_healthy:
                    # Reset recovery tracking on successful check
                    self._reset_strategy_tracking(strategy)
                else:
                    # Attempt recovery
                    self._attempt_component_recovery(name, strategy)
                    
            except Exception as e:
                logger.error(f"Error processing healing strategy for {name}: {str(e)}")

    def _attempt_component_recovery(self, name: str, strategy: Dict[str, Any]) -> None:
        """
        Attempt to recover a specific system component.

        Implements progressive recovery with exponential backoff 
        and administrative notification for persistent failures.

        Args:
            name (str): Name of the component being recovered
            strategy (Dict[str, Any]): Recovery strategy configuration
        """
        strategy["attempts"] += 1
        
        # Calculate recovery delay with exponential backoff
        delay = strategy["initial_delay"] * (strategy["backoff_factor"] ** (strategy["attempts"] - 1))
        time.sleep(delay)
        
        # Attempt healing
        if strategy["attempts"] <= strategy["max_attempts"]:
            healed = strategy["heal"]()
            
            if healed:
                logger.info(f"Successfully recovered {name}")
                strategy["attempts"] = max(0, strategy["attempts"] - 1)
            else:
                self._handle_recovery_failure(name, strategy)
        else:
            self._handle_persistent_failure(name, strategy)

    def _handle_recovery_failure(self, name: str, strategy: Dict[str, Any]) -> None:
        """
        Process individual component recovery failure.

        Args:
            name (str): Name of the failed component
            strategy (Dict[str, Any]): Recovery strategy configuration
        """
        logger.error(f"Failed to heal {name}")
        
        if strategy["severity"] == "critical" and strategy["attempts"] == strategy["max_attempts"]:
            self._send_critical_alert(name, strategy)

    def _handle_persistent_failure(self, name: str, strategy: Dict[str, Any]) -> None:
        """
        Manage components that fail recovery multiple times.

        Args:
            name (str): Name of the persistently failing component
            strategy (Dict[str, Any]): Recovery strategy configuration
        """
        logger.critical(f"{name} has failed recovery attempts, manual intervention required")
        self._send_critical_alert(name, strategy, manual_intervention=True)

    def _send_critical_alert(
        self, 
        component_name: str, 
        strategy: Dict[str, Any], 
        manual_intervention: bool = False
    ) -> None:
        """
        Send critical alerts for system failures.

        Args:
            component_name (str): Name of the failed component
            strategy (Dict[str, Any]): Recovery strategy configuration
            manual_intervention (bool): Indicates if manual intervention is needed
        """
        if self.notification_manager:
            message = (
                f"{component_name} has {'exceeded' if manual_intervention else 'failed'} "
                f"recovery attempts. {'Immediate manual intervention required.' if manual_intervention else ''}"
            )
            
            try:
                self.notification_manager.send_admin_alert(
                    subject=f"Critical System Failure: {component_name}",
                    message=message
                )
            except Exception as e:
                logger.error(f"Failed to send critical alert: {str(e)}")

    def _reset_strategy_tracking(self, strategy: Dict[str, Any]) -> None:
        """
        Reset recovery tracking for a successful component.

        Args:
            strategy (Dict[str, Any]): Recovery strategy configuration
        """
        strategy["attempts"] = 0

    def _check_database_connection(self) -> bool:
        """
        Verify database connection health.

        Returns:
            bool: Indicates database connection status
        """
        try:
            # Check if trading engine exists and has db property
            if self.has_trading_engine and hasattr(self.trading_engine, 'db'):
                db = self.trading_engine.db
                if db and hasattr(db, 'test_connection'):
                    return db.test_connection()
                    
            # General check for any other databases in the component
            for component_name in ['db', 'database', 'database_manager']:
                if hasattr(self, component_name):
                    component = getattr(self, component_name)
                    if hasattr(component, 'test_connection'):
                        return component.test_connection()
                        
            # If no database found to test, assume it's working
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {str(e)}")
            return False

    def _heal_database_connection(self) -> bool:
        """
        Attempt to recover database connection.

        Returns:
            bool: Indicates successful connection recovery
        """
        try:
            logger.info("Attempting database connection recovery")
            
            # Check if trading engine exists and has db property
            if self.has_trading_engine and hasattr(self.trading_engine, 'db'):
                db = self.trading_engine.db
                if db:
                    # Try reconnecting the database
                    if hasattr(db, 'close') and hasattr(db, 'initialize'):
                        try:
                            db.close()
                            time.sleep(2)  # Give time for connections to fully close
                            return db.initialize()
                        except Exception as db_e:
                            logger.error(f"Database reconnection error: {str(db_e)}")
                    
                    # Try alternative reconnection methods
                    if hasattr(db, 'reconnect'):
                        return db.reconnect()
                        
            return False
        except Exception as e:
            logger.error(f"Database connection recovery failed: {str(e)}")
            return False

    def _check_api_connections(self) -> bool:
        """
        Verify API connection health.

        Returns:
            bool: Indicates API connection status
        """
        try:
            # Check trading engine user exchange connections if possible
            if self.has_trading_engine and hasattr(self.trading_engine, 'user_exchange_map'):
                # Check at least one exchange connection
                for user_id, exchange in self.trading_engine.user_exchange_map.items():
                    if exchange and hasattr(exchange, 'check_connection'):
                        if not exchange.check_connection():
                            return False
            
            # All connections ok, or no connections to check
            return True
        except Exception as e:
            logger.error(f"API connection check failed: {str(e)}")
            return False

    def _heal_api_connections(self) -> bool:
        """
        Attempt to recover API connections.

        Returns:
            bool: Indicates successful API connection recovery
        """
        try:
            logger.info("Attempting API connection recovery")
            
            # Check if trading engine exists and has exchange map
            if self.has_trading_engine and hasattr(self.trading_engine, 'user_exchange_map'):
                success = True
                
                # Try to recover each exchange connection
                for user_id, exchange in list(self.trading_engine.user_exchange_map.items()):
                    if exchange:
                        # Try reconnect method if available
                        if hasattr(exchange, 'reconnect'):
                            if not exchange.reconnect():
                                success = False
                        # Otherwise try closing and reinitializing
                        elif hasattr(exchange, 'close') and hasattr(exchange, 'initialize'):
                            try:
                                exchange.close()
                                if not exchange.initialize():
                                    success = False
                            except Exception:
                                success = False
                                
                return success
            
            return False
        except Exception as e:
            logger.error(f"API connection recovery failed: {str(e)}")
            return False

    def _check_trading_engine(self) -> bool:
        """
        Verify trading engine operational status with improved null safety.

        Returns:
            bool: Indicates trading engine health
        """
        try:
            # Check if trading engine exists
            if not self.has_trading_engine or self.trading_engine is None:
                logger.error("Trading engine reference is null or not initialized")
                return False
                
            # Check if it has 'running' attribute
            if not hasattr(self.trading_engine, 'running'):
                logger.error("Trading engine missing 'running' attribute")
                return False
                
            # Return running state
            return self.trading_engine.running
        except Exception as e:
            logger.error(f"Trading engine check failed: {str(e)}")
            return False

    def _heal_trading_engine(self) -> bool:
        """
        Attempt to recover trading engine functionality with enhanced safety.

        Returns:
            bool: Indicates successful trading engine recovery
        """
        try:
            logger.info("Attempting trading engine recovery")
            
            # Verify trading engine exists
            if not self.has_trading_engine or self.trading_engine is None:
                logger.error("Cannot heal null trading engine")
                return False
            
            # Check for required methods
            has_stop = hasattr(self.trading_engine, 'stop') and callable(getattr(self.trading_engine, 'stop'))
            has_start = hasattr(self.trading_engine, 'start') and callable(getattr(self.trading_engine, 'start'))
            
            if not (has_stop and has_start):
                logger.error("Trading engine missing required stop/start methods")
                return False
            
            # Attempt to stop if it's running
            try:
                if hasattr(self.trading_engine, 'running') and self.trading_engine.running:
                    self.trading_engine.stop()
                    logger.info("Successfully stopped trading engine")
                    
                # Give time for full shutdown
                time.sleep(3)
            except Exception as stop_e:
                logger.error(f"Error stopping trading engine: {str(stop_e)}")
            
            # Attempt to start
            try:
                self.trading_engine.start()
                
                # Verify running state
                if hasattr(self.trading_engine, 'running'):
                    running = self.trading_engine.running
                    if running:
                        logger.info("Trading engine successfully restarted")
                    else:
                        logger.error("Trading engine failed to start (not running)")
                    return running
                
                # Can't verify state - assume success
                logger.info("Trading engine restart attempted (status unverifiable)")
                return True
                
            except Exception as start_e:
                logger.error(f"Error starting trading engine: {str(start_e)}")
                return False
                
        except Exception as e:
            logger.error(f"Trading engine recovery failed: {str(e)}")
            return False

    def heal_component(self, name: str, component: Any) -> bool:
        """
        Heal a specific component by name and reference.
        
        This method determines the appropriate healing strategy for the component
        and attempts to recover it based on its type and state.
        
        Args:
            name (str): Name identifier of the component to heal
            component (Any): The component instance to heal
            
        Returns:
            bool: True if healing was successful, False otherwise
        """
        logger.info(f"Attempting to heal component: {name}")
        
        # Handle null component
        if component is None:
            logger.error(f"Cannot heal null component: {name}")
            return False
            
        # Track component recovery history
        if name not in self.recovery_history:
            self.recovery_history[name] = {
                "attempts": 0,
                "last_attempt": time.time(),
                "success_count": 0,
                "failure_count": 0
            }
        
        history = self.recovery_history[name]
        history["attempts"] += 1
        history["last_attempt"] = time.time()
        
        # Determine component type and appropriate strategy
        if hasattr(component, "start") and callable(getattr(component, "start")):
            return self._restart_component(name, component, history)
        elif hasattr(component, "reconnect") and callable(getattr(component, "reconnect")):
            return self._reconnect_component(name, component, history)
        elif hasattr(component, "initialize") and callable(getattr(component, "initialize")):
            return self._reinitialize_component(name, component, history)
        else:
            # No known recovery method
            logger.error(f"No healing strategy found for component {name}")
            history["failure_count"] += 1
            return False

    def _restart_component(self, name: str, component: Any, history: Dict[str, Any]) -> bool:
        """
        Restart a component that has start/stop methods.
        
        Args:
            name (str): Component name
            component (Any): Component instance
            history (Dict[str, Any]): Recovery history for this component
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Restarting component: {name}")
            
            # Check if component has a stop method
            if hasattr(component, "stop") and callable(getattr(component, "stop")):
                try:
                    component.stop()
                    logger.info(f"Successfully stopped component: {name}")
                except Exception as e:
                    logger.warning(f"Error stopping component {name}: {str(e)}")
            
            # Wait briefly before restarting
            time.sleep(2)
            
            # Start the component
            component.start()
            
            # Verify component is running if it has a 'running' attribute
            if hasattr(component, "running"):
                if component.running:
                    logger.info(f"Successfully restarted component: {name}")
                    history["success_count"] += 1
                    return True
                else:
                    logger.error(f"Component {name} failed to start (not running)")
                    history["failure_count"] += 1
                    return False
            else:
                # Assume success if we can't verify
                logger.info(f"Restarted component: {name} (status unverifiable)")
                history["success_count"] += 1
                return True
                
        except Exception as e:
            logger.error(f"Failed to restart component {name}: {str(e)}")
            history["failure_count"] += 1
            
            # Send notification about component failure
            if self.has_notification_manager and self.notification_manager:
                try:
                    self.notification_manager.send_admin_alert(
                        f"Component Restart Failure: {name}",
                        f"Failed to restart component {name}: {str(e)}"
                    )
                except Exception as notify_e:
                    logger.error(f"Failed to send restart failure notification: {str(notify_e)}")
                    
            return False

    def _reconnect_component(self, name: str, component: Any, history: Dict[str, Any]) -> bool:
        """
        Reconnect a component that has a reconnect method.
        
        Args:
            name (str): Component name
            component (Any): Component instance
            history (Dict[str, Any]): Recovery history for this component
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Reconnecting component: {name}")
            result = component.reconnect()
            
            if result:
                logger.info(f"Successfully reconnected component: {name}")
                history["success_count"] += 1
                return True
            else:
                logger.error(f"Failed to reconnect component: {name}")
                history["failure_count"] += 1
                return False
                
        except Exception as e:
            logger.error(f"Error reconnecting component {name}: {str(e)}")
            history["failure_count"] += 1
            return False

    def _reinitialize_component(self, name: str, component: Any, history: Dict[str, Any]) -> bool:
        """
        Reinitialize a component that has an initialize method.
        
        Args:
            name (str): Component name
            component (Any): Component instance
            history (Dict[str, Any]): Recovery history for this component
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Reinitializing component: {name}")
            result = component.initialize()
            
            if result:
                logger.info(f"Successfully reinitialized component: {name}")
                history["success_count"] += 1
                return True
            else:
                logger.error(f"Failed to reinitialize component: {name}")
                history["failure_count"] += 1
                return False
                
        except Exception as e:
            logger.error(f"Error reinitializing component {name}: {str(e)}")
            history["failure_count"] += 1
            return False
    
    def get_recovery_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get recovery statistics for all components.
        
        Returns:
            Dict mapping component names to their recovery statistics
        """
        return self.recovery_history
        
    def update_component(self, name: str, component: Any) -> bool:
        """
        Update a component reference that may have been reinitialized elsewhere.
        
        Args:
            name (str): Component name (e.g., 'trading_engine')
            component (Any): New component instance
            
        Returns:
            bool: Success status
        """
        try:
            # Update the component reference
            if name == 'trading_engine':
                self.trading_engine = component
                self.has_trading_engine = component is not None
                
                # Update healing strategies if necessary
                if component is not None and 'trading_engine' not in self.healing_strategies:
                    self.healing_strategies["trading_engine"] = {
                        "check": self._check_trading_engine,
                        "heal": self._heal_trading_engine,
                        "severity": "critical",
                        "attempts": 0,
                        "max_attempts": 3,
                        "backoff_factor": 2.0,
                        "initial_delay": 10.0
                    }
                elif component is None and 'trading_engine' in self.healing_strategies:
                    del self.healing_strategies['trading_engine']
                    
            elif name == 'system_monitor':
                self.system_monitor = component
                self.has_system_monitor = component is not None
                
            elif name == 'notification_manager':
                self.notification_manager = component
                self.has_notification_manager = component is not None
                
            else:
                logger.warning(f"Unknown component type: {name}")
                return False
                
            logger.info(f"Component {name} reference updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating component {name}: {str(e)}")
            return False

# Maintain backward compatibility
SelfHealing = SelfHealingSystem