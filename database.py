"""
Database module for managing stock data storage in SQLite.

This module handles:
- Creating database tables
- Storing stock data
- Retrieving stock data
- Database connection management
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime


class StockDatabase:
    """Class to handle all database operations for stock data."""
    
    def __init__(self, db_path="data/stocks.db"):
        """
        Initialize the database connection.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.create_tables()
    
    def create_tables(self):
        """Create the necessary tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if frequency column exists, if not add it
            cursor.execute("PRAGMA table_info(stock_data)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Create table for stock data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    frequency TEXT NOT NULL DEFAULT 'daily',
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date, frequency)
                )
            ''')
            
            # Add frequency column if it doesn't exist (for existing databases)
            if 'frequency' not in columns:
                cursor.execute('ALTER TABLE stock_data ADD COLUMN frequency TEXT DEFAULT "daily"')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_ticker_date_freq 
                ON stock_data(ticker, date, frequency)
            ''')
            
            # Create table for stock info (company details that don't change often)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_info (
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap INTEGER,
                    pe_ratio REAL,
                    dividend_yield REAL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def insert_stock_data(self, ticker, data_df, frequency='daily'):
        """
        Insert stock data into the database.
        
        Args:
            ticker (str): Stock ticker symbol
            data_df (pd.DataFrame): DataFrame with stock data
            frequency (str): Data frequency ('daily' or 'weekly')
        """
        with sqlite3.connect(self.db_path) as conn:
            # Prepare data for insertion  
            data_df_copy = data_df.copy()
            data_df_copy['ticker'] = ticker
            data_df_copy['date'] = data_df_copy.index
            
            # Handle different possible column names from yfinance
            column_mapping = {}
            available_columns = data_df_copy.columns.tolist()
            
            # Map yfinance columns to our database schema
            for col in available_columns:
                col_lower = col.lower()
                if col_lower in ['open']:
                    column_mapping['open'] = col
                elif col_lower in ['high']:
                    column_mapping['high'] = col
                elif col_lower in ['low']:
                    column_mapping['low'] = col
                elif col_lower in ['close']:
                    column_mapping['close'] = col
                elif col_lower in ['adj close', 'adj_close', 'adjclose']:
                    column_mapping['adj_close'] = col
                elif col_lower in ['volume']:
                    column_mapping['volume'] = col
            
            # Create the final dataframe with standardized columns
            final_data = pd.DataFrame()
            final_data['ticker'] = data_df_copy['ticker']
            final_data['date'] = data_df_copy['date']
            final_data['frequency'] = frequency
            
            # Add price and volume columns if they exist
            for db_col, df_col in column_mapping.items():
                if df_col in data_df_copy.columns:
                    final_data[db_col] = data_df_copy[df_col]
                else:
                    final_data[db_col] = None
            
            # Insert data, replacing duplicates using INSERT OR IGNORE
            cursor = conn.cursor()
            for _, row in final_data.iterrows():
                # Convert pandas/numpy types to Python native types and handle NaN values
                def clean_value(value):
                    if pd.isna(value):
                        return None
                    # Convert numpy types to Python native types
                    if hasattr(value, 'item'):
                        return value.item()
                    return value
                
                # Convert date to string format
                date_str = row['date']
                if hasattr(date_str, 'strftime'):
                    date_str = date_str.strftime('%Y-%m-%d')
                elif hasattr(date_str, 'date'):
                    date_str = str(date_str.date())
                else:
                    date_str = str(date_str)
                
                cursor.execute('''
                    INSERT OR IGNORE INTO stock_data 
                    (ticker, date, frequency, open, high, low, close, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(row['ticker']), 
                    date_str, 
                    str(row['frequency']),
                    clean_value(row.get('open')), 
                    clean_value(row.get('high')), 
                    clean_value(row.get('low')), 
                    clean_value(row.get('close')), 
                    clean_value(row.get('adj_close')), 
                    clean_value(row.get('volume'))
                ))
            conn.commit()
    
    def get_stock_data(self, tickers, start_date=None, end_date=None, frequency='daily'):
        """
        Retrieve stock data from the database.
        
        Args:
            tickers (list): List of stock ticker symbols
            start_date (str, optional): Start date for data retrieval
            end_date (str, optional): End date for data retrieval
            frequency (str): Data frequency ('daily' or 'weekly')
            
        Returns:
            pd.DataFrame: DataFrame with stock data
        """
        with sqlite3.connect(self.db_path) as conn:
            # Build query
            placeholders = ','.join(['?' for _ in tickers])
            query = f'''
                SELECT ticker, date, open, high, low, close, adj_close, volume
                FROM stock_data
                WHERE ticker IN ({placeholders}) AND frequency = ?
            '''
            
            params = list(tickers) + [frequency]
            
            if start_date:
                query += ' AND date >= ?'
                params.append(start_date)
            
            if end_date:
                query += ' AND date <= ?'
                params.append(end_date)
            
            query += ' ORDER BY ticker, date'
            
            # Execute query and return DataFrame
            df = pd.read_sql_query(query, conn, params=params)
            
            # Handle timezone-aware date parsing - ensure consistent timezone-naive format
            if not df.empty:
                try:
                    # Parse dates and handle mixed formats with UTC to avoid warnings
                    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert(None)
                except Exception:
                    try:
                        # Fallback: parse with UTC=True to handle mixed timezones
                        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
                    except Exception:
                        try:
                            # Fallback: parse without timezone
                            df['date'] = pd.to_datetime(df['date'])
                            # Remove timezone if present
                            if hasattr(df['date'].dt, 'tz') and df['date'].dt.tz is not None:
                                df['date'] = df['date'].dt.tz_localize(None)
                        except Exception:
                            # Final fallback: coerce errors to NaT and drop
                            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.tz_localize(None)
                            df = df.dropna(subset=['date'])
            return df
    
    def get_available_tickers(self):
        """
        Get list of all tickers available in the database.
        
        Returns:
            list: List of ticker symbols
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT ticker FROM stock_data ORDER BY ticker')
            return [row[0] for row in cursor.fetchall()]
    
    def delete_stock_data(self, ticker, frequency=None):
        """
        Delete data for a specific ticker and optionally specific frequency.
        
        Args:
            ticker (str): Stock ticker symbol to delete
            frequency (str, optional): Data frequency ('daily' or 'weekly'). If None, deletes all frequencies.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if frequency:
                cursor.execute('DELETE FROM stock_data WHERE ticker = ? AND frequency = ?', (ticker, frequency))
            else:
                cursor.execute('DELETE FROM stock_data WHERE ticker = ?', (ticker,))
            conn.commit()
    
    def get_data_date_range(self, ticker, frequency='daily'):
        """
        Get the date range of available data for a ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            frequency (str): Data frequency ('daily' or 'weekly')
            
        Returns:
            tuple: (start_date, end_date) or (None, None) if no data
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT MIN(date), MAX(date) 
                FROM stock_data 
                WHERE ticker = ? AND frequency = ?
            ''', (ticker, frequency))
            
            result = cursor.fetchone()
            return result if result and result[0] else (None, None)
    
    def check_data_freshness(self, ticker, start_date, end_date, frequency='daily'):
        """
        Check if we have fresh data for a ticker in the requested date range.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Requested start date
            end_date (str): Requested end date
            frequency (str): Data frequency ('daily' or 'weekly')
            
        Returns:
            dict: Information about data availability and freshness
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check what data we have
            cursor.execute('''
                SELECT MIN(date), MAX(date), COUNT(*) 
                FROM stock_data 
                WHERE ticker = ? AND frequency = ? AND date BETWEEN ? AND ?
            ''', (ticker, frequency, start_date, end_date))
            
            result = cursor.fetchone()
            
            if not result or not result[0]:
                return {
                    'has_data': False,
                    'needs_fetch': True,
                    'fetch_reason': 'No data found in database'
                }
            
            db_start, db_end, count = result
            
            # Check if we have recent data (within last 24 hours for current date)
            from datetime import datetime, timedelta
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            # If requesting current data, check if we have yesterday's data
            if end_date >= str(today) and db_end >= str(yesterday):
                return {
                    'has_data': True,
                    'needs_fetch': False,
                    'fetch_reason': 'Recent data available',
                    'data_range': (db_start, db_end),
                    'record_count': count
                }
            
            # Check if we have complete coverage for requested range
            expected_days = self._calculate_trading_days(start_date, end_date)
            coverage_ratio = count / max(expected_days, 1)
            
            if coverage_ratio >= 0.8:  # 80% coverage is acceptable
                return {
                    'has_data': True,
                    'needs_fetch': False,
                    'fetch_reason': f'Good coverage ({coverage_ratio:.1%})',
                    'data_range': (db_start, db_end),
                    'record_count': count
                }
            else:
                return {
                    'has_data': True,
                    'needs_fetch': True,
                    'fetch_reason': f'Incomplete data ({coverage_ratio:.1%} coverage)',
                    'data_range': (db_start, db_end),
                    'record_count': count
                }
    
    def _calculate_trading_days(self, start_date, end_date):
        """
        Rough calculation of trading days (excluding weekends).
        
        Args:
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            int: Approximate number of trading days
        """
        from datetime import datetime
        
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            total_days = (end - start).days + 1
            
            # Rough estimate: 5/7 of days are trading days
            return int(total_days * 5 / 7)
        except:
            return 250  # Default to about 1 year of trading days
    
    def get_database_stats(self):
        """
        Get statistics about the database size and usage.
        
        Returns:
            dict: Database statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get total record count
            cursor.execute('SELECT COUNT(*) FROM stock_data')
            total_records = cursor.fetchone()[0]
            
            # Get unique tickers count
            cursor.execute('SELECT COUNT(DISTINCT ticker) FROM stock_data')
            unique_tickers = cursor.fetchone()[0]
            
            # Get date range
            cursor.execute('SELECT MIN(date), MAX(date) FROM stock_data')
            date_range = cursor.fetchone()
            
            # Get database file size
            import os
            db_size_bytes = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            db_size_mb = db_size_bytes / (1024 * 1024)
            
            return {
                'total_records': total_records,
                'unique_tickers': unique_tickers,
                'date_range': date_range,
                'size_mb': db_size_mb,
                'size_bytes': db_size_bytes
            }
    
    def cleanup_old_data(self, max_size_mb=50, max_age_days=180):
        """
        Automatically clean up old data to keep database size manageable.
        
        Args:
            max_size_mb (int): Maximum database size in MB before cleanup
            max_age_days (int): Maximum age of data to keep in days
        """
        stats = self.get_database_stats()
        
        # Check if cleanup is needed
        needs_cleanup = (
            stats['size_mb'] > max_size_mb or 
            stats['total_records'] > 50000  # Also cleanup if too many records
        )
        
        if not needs_cleanup:
            return False  # No cleanup needed
        
        from datetime import datetime, timedelta
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Calculate cutoff date
            cutoff_date = (datetime.now() - timedelta(days=max_age_days)).strftime('%Y-%m-%d')
            
            # First, try deleting very old data (older than max_age_days)
            cursor.execute('DELETE FROM stock_data WHERE date < ?', (cutoff_date,))
            deleted_old = cursor.rowcount
            
            # If still too big, delete data for tickers that haven't been accessed recently
            if stats['size_mb'] > max_size_mb:
                # Keep only the most recent 90 days of data for each ticker
                recent_cutoff = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                cursor.execute('DELETE FROM stock_data WHERE date < ?', (recent_cutoff,))
                deleted_recent = cursor.rowcount
            else:
                deleted_recent = 0
            
            # If still too big, keep only top 20 most common tickers
            if stats['size_mb'] > max_size_mb:
                cursor.execute('''
                    DELETE FROM stock_data 
                    WHERE ticker NOT IN (
                        SELECT ticker 
                        FROM (
                            SELECT ticker, COUNT(*) as count 
                            FROM stock_data 
                            GROUP BY ticker 
                            ORDER BY count DESC 
                            LIMIT 20
                        )
                    )
                ''')
                deleted_uncommon = cursor.rowcount
            else:
                deleted_uncommon = 0
            
            conn.commit()
            
            total_deleted = deleted_old + deleted_recent + deleted_uncommon
            
            return {
                'cleanup_performed': True,
                'deleted_old': deleted_old,
                'deleted_recent': deleted_recent, 
                'deleted_uncommon': deleted_uncommon,
                'total_deleted': total_deleted,
                'cutoff_date': cutoff_date
            }
    
    def update_access_time(self, ticker):
        """
        Update the last access time for a ticker (for cleanup prioritization).
        This could be expanded to track usage patterns.
        
        Args:
            ticker (str): Stock ticker symbol
        """
        # For now, we'll use the created_at field that already exists
        # In a more sophisticated system, we'd add a last_accessed column
        pass
    
    def store_stock_info(self, ticker, stock_info):
        """
        Store stock information (company details) in the database.
        
        Args:
            ticker (str): Stock ticker symbol
            stock_info (dict): Dictionary with stock information
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clean values and handle None/NaN
            def clean_value(value):
                if pd.isna(value) or value == 'N/A':
                    return None
                if hasattr(value, 'item'):
                    return value.item()
                return value
            
            cursor.execute('''
                INSERT OR REPLACE INTO stock_info 
                (ticker, name, sector, industry, market_cap, pe_ratio, dividend_yield, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                ticker,
                clean_value(stock_info.get('name')),
                clean_value(stock_info.get('sector')),
                clean_value(stock_info.get('industry')),
                clean_value(stock_info.get('market_cap')),
                clean_value(stock_info.get('pe_ratio')),
                clean_value(stock_info.get('dividend_yield'))
            ))
            conn.commit()
    
    def get_cached_stock_info(self, ticker):
        """
        Retrieve stock information from the database.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Dictionary with stock information, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT name, sector, industry, market_cap, pe_ratio, dividend_yield, last_updated
                FROM stock_info
                WHERE ticker = ?
            ''', (ticker,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'name': result[0] or ticker,
                    'sector': result[1] or 'N/A',
                    'industry': result[2] or 'N/A',
                    'market_cap': result[3] or 'N/A',
                    'pe_ratio': result[4] or 'N/A',
                    'dividend_yield': result[5] or 'N/A',
                    'last_updated': result[6]
                }
            return None
    
    def is_stock_info_fresh(self, ticker, max_age_days=30):
        """
        Check if stock info is fresh (not older than max_age_days).
        
        Args:
            ticker (str): Stock ticker symbol
            max_age_days (int): Maximum age in days
            
        Returns:
            bool: True if fresh, False if stale or not found
        """
        stock_info = self.get_cached_stock_info(ticker)
        if not stock_info:
            return False
        
        try:
            from datetime import datetime, timedelta
            last_updated = datetime.strptime(stock_info['last_updated'], '%Y-%m-%d %H:%M:%S')
            return (datetime.now() - last_updated).days < max_age_days
        except:
            return False