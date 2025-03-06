# core/scheduler.py
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, Callable, Union, Optional

logger = logging.getLogger(__name__)

class Scheduler:
    """
    Advanced Task Scheduling System

    Provides a comprehensive scheduling mechanism for managing 
    periodic jobs with precise execution controls and robust error handling.
    """

    def __init__(
        self, 
        shutdown_event: Optional[threading.Event] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Scheduler with configurable dependencies.

        Args:
            shutdown_event (threading.Event, optional): Event to signal system shutdown
            config (dict, optional): Configuration settings for scheduling
        """
        # Shutdown management
        self.shutdown_event = shutdown_event or threading.Event()
        
        # Job management
        self.jobs: Dict[str, Dict[str, Any]] = {}
        
        # Threading controls
        self.thread: Optional[threading.Thread] = None
        self.running = False
        
        # Configuration
        self.config = config or {}
        
        logger.info("Scheduler initialized successfully")
        
    def start(self):
        """
        Initiate the scheduler's background monitoring thread.

        Activates periodic job execution with comprehensive error handling.
        """
        if self.running:
            logger.warning("Scheduler is already operational")
            return
            
        logger.info("Activating scheduler")
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
    def stop(self):
        """
        Gracefully terminate the scheduler.

        Ensures all ongoing jobs are completed and resources are released.
        """
        if not self.running:
            logger.warning("Scheduler is already inactive")
            return
            
        logger.info("Deactivating scheduler")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=5)
            
    def _run(self):
        """
        Primary scheduling loop for job execution.

        Continuously monitors and executes scheduled jobs 
        while respecting system shutdown signals.
        """
        while self.running and not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                
                # Check and execute due jobs
                for job_id, job in list(self.jobs.items()):
                    if self._should_run_job(job, current_time):
                        self._execute_job(job)
                
                # Prevent tight looping
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Critical error in scheduler loop: {str(e)}")
                time.sleep(5)
    
    def _should_run_job(self, job: Dict[str, Any], now: datetime) -> bool:
        """
        Determine if a scheduled job should be executed.

        Args:
            job (Dict): Job configuration details
            now (datetime): Current timestamp

        Returns:
            bool: Indicates whether the job should be run
        """
        job_type = job['type']
        
        if job_type == 'interval':
            return (now.timestamp() - job['last_run']) >= job['interval']
        
        if job_type == 'daily':
            return (now.hour == job['hour'] and 
                    now.minute == job['minute'] and 
                    now.second < 60)
        
        if job_type == 'weekly':
            return (now.weekday() == job['day_of_week'] and 
                    now.hour == job['hour'] and 
                    now.minute == job['minute'] and 
                    now.second < 60)
        
        if job_type == 'monthly':
            return (now.day == job['day'] and 
                    now.hour == job['hour'] and 
                    now.minute == job['minute'] and 
                    now.second < 60)
        
        return False
    
    def _execute_job(self, job: Dict[str, Any]):
        """
        Execute a scheduled job with comprehensive error management.

        Args:
            job (Dict): Job configuration to execute
        """
        try:
            # Update last run timestamp
            job['last_run'] = datetime.now().timestamp()
            
            # Execute job function
            job['func'](*job['args'], **job['kwargs'])
            
            logger.debug(f"Executed scheduled job: {job['id']}")
            
        except Exception as e:
            logger.error(f"Job execution failed: {job['id']} - {str(e)}")
    
    def schedule(
        self,
        func: Callable,
        interval_seconds: Union[int, float],
        task_id: str,
        run_at_time: Optional[str] = None,
        *args: Any,
        **kwargs: Any
    ):
        """
        Schedule a task to run at specified intervals.
        Compatibility method that maps to the appropriate underlying implementation.
        
        Args:
            func: Function to execute
            interval_seconds: Interval between executions in seconds
            task_id: Unique identifier for the task
            run_at_time: Optional specific time to run the task (format: "HH:MM:SS")
        """
        if run_at_time:
            # Parse time string to get hour and minute
            try:
                time_parts = run_at_time.split(':')
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                return self.add_daily_job(task_id, func, hour, minute, *args, **kwargs)
            except (ValueError, IndexError) as e:
                logger.error(f"Invalid time format for run_at_time: {run_at_time}. Using interval scheduling instead.")
                return self.add_job(task_id, func, interval_seconds, *args, **kwargs)
        else:
            # Use interval scheduling
            return self.add_job(task_id, func, interval_seconds, *args, **kwargs)
    
    def add_job(
        self, 
        job_id: str, 
        func: Callable, 
        interval: Union[int, float], 
        *args: Any, 
        **kwargs: Any
    ):
        """
        Schedule a recurring job with fixed interval execution.

        Args:
            job_id (str): Unique identifier for the job
            func (Callable): Function to be executed
            interval (int/float): Execution interval in seconds
        """
        self.jobs[job_id] = {
            'id': job_id,
            'func': func,
            'type': 'interval',
            'interval': interval,
            'last_run': 0,  # Immediate first run
            'args': args,
            'kwargs': kwargs
        }
        
        logger.info(f"Interval job added: {job_id}, {interval}s")
    
    def add_daily_job(
        self, 
        job_id: str, 
        func: Callable, 
        hour: int, 
        minute: int, 
        *args: Any, 
        **kwargs: Any
    ):
        """
        Schedule a daily recurring job at a specific time.

        Args:
            job_id (str): Unique identifier for the job
            func (Callable): Function to be executed
            hour (int): Hour of execution (0-23)
            minute (int): Minute of execution (0-59)
        """
        self.jobs[job_id] = {
            'id': job_id,
            'func': func,
            'type': 'daily',
            'hour': hour,
            'minute': minute,
            'last_run': 0,
            'args': args,
            'kwargs': kwargs
        }
        
        logger.info(f"Daily job added: {job_id}, {hour:02d}:{minute:02d}")
    
    def add_weekly_job(
        self, 
        job_id: str, 
        func: Callable, 
        day_of_week: Union[int, str], 
        hour: int, 
        minute: int, 
        *args: Any, 
        **kwargs: Any
    ):
        """
        Schedule a weekly recurring job at a specific time.

        Args:
            job_id (str): Unique identifier for the job
            func (Callable): Function to be executed
            day_of_week (int/str): Day of week (0-6 or name)
            hour (int): Hour of execution (0-23)
            minute (int): Minute of execution (0-59)
        """
        day_mapping = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        if isinstance(day_of_week, str):
            day_of_week = day_mapping.get(day_of_week.lower(), 0)
        
        self.jobs[job_id] = {
            'id': job_id,
            'func': func,
            'type': 'weekly',
            'day_of_week': day_of_week,
            'hour': hour,
            'minute': minute,
            'last_run': 0,
            'args': args,
            'kwargs': kwargs
        }
        
        logger.info(f"Weekly job added: {job_id}, Day {day_of_week}, {hour:02d}:{minute:02d}")
    
    def add_monthly_job(
        self, 
        job_id: str, 
        func: Callable, 
        day: int, 
        hour: int, 
        minute: int, 
        *args: Any, 
        **kwargs: Any
    ):
        """
        Schedule a monthly recurring job at a specific time.

        Args:
            job_id (str): Unique identifier for the job
            func (Callable): Function to be executed
            day (int): Day of month (1-31)
            hour (int): Hour of execution (0-23)
            minute (int): Minute of execution (0-59)
        """
        self.jobs[job_id] = {
            'id': job_id,
            'func': func,
            'type': 'monthly',
            'day': day,
            'hour': hour,
            'minute': minute,
            'last_run': 0,
            'args': args,
            'kwargs': kwargs
        }
        
        logger.info(f"Monthly job added: {job_id}, Day {day}, {hour:02d}:{minute:02d}")
    
    def remove_job(self, job_id: str):
        """
        Remove a scheduled job.

        Args:
            job_id (str): Unique identifier of the job to remove
        """
        if job_id in self.jobs:
            del self.jobs[job_id]
            logger.info(f"Job removed: {job_id}")
        else:
            logger.warning(f"Job not found: {job_id}")

# Maintain backward compatibility
TaskScheduler = Scheduler