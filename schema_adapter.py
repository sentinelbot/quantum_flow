# database/utils/schema_adapter.py
import logging
from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)

class SchemaAdapter:
    """
    Utility class to handle schema differences and enable graceful operation
    with incomplete database schemas.
    """
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.column_cache = {}
        
    def has_column(self, table_name, column_name):
        """
        Check if a column exists in a table
        
        Args:
            table_name: Name of the table to check
            column_name: Name of the column to look for
            
        Returns:
            bool: True if column exists, False otherwise
        """
        cache_key = f"{table_name}.{column_name}"
        
        # Return cached result if available
        if cache_key in self.column_cache:
            return self.column_cache[cache_key]
            
        # Check database schema
        try:
            session = self.db.get_session()
            inspector = inspect(self.db.engine)
            columns = [col['name'] for col in inspector.get_columns(table_name)]
            result = column_name in columns
            self.column_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Error checking column existence: {str(e)}")
            return False
        finally:
            session.close()
            
    def safe_query(self, model_class, filters=None):
        """
        Build a query that works regardless of schema differences
        
        Args:
            model_class: SQLAlchemy model class
            filters: Dictionary of filters to apply
            
        Returns:
            SQLAlchemy query object or None if error
        """
        try:
            session = self.db.get_session()
            query = session.query(model_class)
            
            if filters:
                for column, value in filters.items():
                    # Check if column exists before filtering on it
                    if self.has_column(model_class.__tablename__, column):
                        query = query.filter(getattr(model_class, column) == value)
            
            return query
        except Exception as e:
            logger.error(f"Error building safe query: {str(e)}")
            return None