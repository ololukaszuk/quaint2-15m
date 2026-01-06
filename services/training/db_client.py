"""
Shared Database Client
PostgreSQL/TimescaleDB connection wrapper with connection pooling
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
import time

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, execute_values

logger = logging.getLogger(__name__)


class DatabaseClient:
    """
    PostgreSQL/TimescaleDB client with connection pooling and error handling.
    
    Features:
    - Connection pooling for performance
    - Automatic retry logic
    - Context managers for safe transactions
    - Type hints for better IDE support
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        min_connections: int = 1,
        max_connections: int = 10
    ):
        """
        Initialize database client with connection pool.
        
        Args:
            host: Database host
            port: Database port
            user: Database user
            password: Database password
            database: Database name
            min_connections: Minimum pool size
            max_connections: Maximum pool size
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        
        # Create connection pool
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                min_connections,
                max_connections,
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                connect_timeout=10
            )
            logger.info(f"Database connection pool created: {host}:{port}/{database}")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    def get_connection(self):
        """
        Get a connection from the pool.
        
        Returns:
            psycopg2 connection object
        """
        try:
            conn = self.pool.getconn()
            return conn
        except Exception as e:
            logger.error(f"Failed to get connection from pool: {e}")
            raise
    
    def return_connection(self, conn):
        """
        Return connection to the pool.
        
        Args:
            conn: Connection to return
        """
        try:
            self.pool.putconn(conn)
        except Exception as e:
            logger.error(f"Failed to return connection to pool: {e}")
    
    @contextmanager
    def get_cursor(self, dict_cursor: bool = False):
        """
        Context manager for database cursor with automatic connection handling.
        
        Args:
            dict_cursor: Use RealDictCursor if True
            
        Yields:
            Database cursor
            
        Example:
            with db.get_cursor() as cur:
                cur.execute("SELECT * FROM table")
                results = cur.fetchall()
        """
        conn = self.get_connection()
        cursor_factory = RealDictCursor if dict_cursor else None
        
        try:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            cursor.close()
            self.return_connection(conn)
    
    def execute(
        self,
        query: str,
        params: Optional[Tuple] = None,
        retry_count: int = 3
    ) -> None:
        """
        Execute a query without returning results (INSERT, UPDATE, DELETE).
        
        Args:
            query: SQL query
            params: Query parameters
            retry_count: Number of retry attempts
        """
        for attempt in range(retry_count):
            try:
                with self.get_cursor() as cur:
                    cur.execute(query, params)
                return
            except Exception as e:
                logger.warning(f"Execute attempt {attempt + 1} failed: {e}")
                if attempt == retry_count - 1:
                    raise
                time.sleep(1)
    
    def fetch_one(
        self,
        query: str,
        params: Optional[Tuple] = None,
        dict_result: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row.
        
        Args:
            query: SQL query
            params: Query parameters
            dict_result: Return as dictionary
            
        Returns:
            Single row as dict or tuple, None if no results
        """
        try:
            with self.get_cursor(dict_cursor=dict_result) as cur:
                cur.execute(query, params)
                result = cur.fetchone()
                return result
        except Exception as e:
            logger.error(f"fetch_one failed: {e}")
            raise
    
    def fetch_many(
        self,
        query: str,
        params: Optional[Tuple] = None,
        dict_result: bool = True,
        size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch multiple rows.
        
        Args:
            query: SQL query
            params: Query parameters
            dict_result: Return as list of dicts
            size: Number of rows to fetch (None = all)
            
        Returns:
            List of rows as dicts or tuples
        """
        try:
            with self.get_cursor(dict_cursor=dict_result) as cur:
                cur.execute(query, params)
                if size:
                    results = cur.fetchmany(size)
                else:
                    results = cur.fetchall()
                return results
        except Exception as e:
            logger.error(f"fetch_many failed: {e}")
            raise
    
    def fetch_all(
        self,
        query: str,
        params: Optional[Tuple] = None,
        dict_result: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch all rows (alias for fetch_many without size).
        
        Args:
            query: SQL query
            params: Query parameters
            dict_result: Return as list of dicts
            
        Returns:
            List of all rows
        """
        return self.fetch_many(query, params, dict_result, size=None)
    
    def insert(
        self,
        table: str,
        data: Dict[str, Any],
        on_conflict: Optional[str] = None
    ) -> None:
        """
        Insert a single row.
        
        Args:
            table: Table name
            data: Dictionary of column: value pairs
            on_conflict: ON CONFLICT clause (e.g., "DO NOTHING")
            
        Example:
            db.insert('ohlcv_raw', {
                'time': '2026-01-06 12:00:00',
                'close': 42000.0
            }, on_conflict='DO NOTHING')
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        values = tuple(data.values())
        
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        if on_conflict:
            query += f" ON CONFLICT {on_conflict}"
        
        self.execute(query, values)
    
    def bulk_insert(
        self,
        table: str,
        data: List[Dict[str, Any]],
        on_conflict: Optional[str] = None,
        batch_size: int = 1000
    ) -> None:
        """
        Bulk insert multiple rows efficiently.
        
        Args:
            table: Table name
            data: List of dictionaries
            on_conflict: ON CONFLICT clause
            batch_size: Number of rows per batch
        """
        if not data:
            return
        
        columns = list(data[0].keys())
        col_str = ', '.join(columns)
        
        query = f"INSERT INTO {table} ({col_str}) VALUES %s"
        if on_conflict:
            query += f" ON CONFLICT {on_conflict}"
        
        # Process in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            values = [tuple(row[col] for col in columns) for row in batch]
            
            try:
                with self.get_cursor() as cur:
                    execute_values(cur, query, values)
            except Exception as e:
                logger.error(f"Bulk insert failed at batch {i}: {e}")
                raise
    
    def update(
        self,
        table: str,
        data: Dict[str, Any],
        where_clause: str,
        where_params: Optional[Tuple] = None
    ) -> None:
        """
        Update rows in table.
        
        Args:
            table: Table name
            data: Dictionary of column: value pairs to update
            where_clause: WHERE condition (without WHERE keyword)
            where_params: Parameters for WHERE clause
            
        Example:
            db.update(
                'predictions',
                {'confidence': 0.75},
                'time = %s',
                ('2026-01-06 12:00:00',)
            )
        """
        set_clause = ', '.join([f"{k} = %s" for k in data.keys()])
        values = tuple(data.values())
        
        query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        params = values + (where_params or ())
        
        self.execute(query, params)
    
    def delete(
        self,
        table: str,
        where_clause: str,
        where_params: Optional[Tuple] = None
    ) -> None:
        """
        Delete rows from table.
        
        Args:
            table: Table name
            where_clause: WHERE condition (without WHERE keyword)
            where_params: Parameters for WHERE clause
        """
        query = f"DELETE FROM {table} WHERE {where_clause}"
        self.execute(query, where_params)
    
    def table_exists(self, table: str) -> bool:
        """
        Check if table exists.
        
        Args:
            table: Table name
            
        Returns:
            True if table exists
        """
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            )
        """
        result = self.fetch_one(query, (table,), dict_result=False)
        return result[0] if result else False
    
    def health_check(self) -> bool:
        """
        Check database connectivity.
        
        Returns:
            True if database is accessible
        """
        try:
            with self.get_cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                return result is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def close(self):
        """Close all connections in pool."""
        if hasattr(self, 'pool') and self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")


def create_db_client_from_env() -> DatabaseClient:
    """
    Create DatabaseClient from environment variables.
    
    Returns:
        Configured DatabaseClient instance
    """
    return DatabaseClient(
        host=os.getenv('DB_HOST', 'timescaledb'),
        port=int(os.getenv('DB_PORT', '5432')),
        user=os.getenv('DB_USER', 'mluser'),
        password=os.getenv('DB_PASSWORD', ''),
        database=os.getenv('DB_NAME', 'btc_ml'),
        min_connections=int(os.getenv('DB_MIN_CONNECTIONS', '1')),
        max_connections=int(os.getenv('DB_MAX_CONNECTIONS', '10'))
    )
