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
            
            # Create table for stock data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date)
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_ticker_date 
                ON stock_data(ticker, date)
            ''')
            
            conn.commit()
    
    def insert_stock_data(self, ticker, data_df):
        """
        Insert stock data into the database.
        
        Args:
            ticker (str): Stock ticker symbol
            data_df (pd.DataFrame): DataFrame with stock data
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
            
            # Add price and volume columns if they exist
            for db_col, df_col in column_mapping.items():
                if df_col in data_df_copy.columns:
                    final_data[db_col] = data_df_copy[df_col]
                else:
                    final_data[db_col] = None
            
            # Insert data, replacing duplicates using INSERT OR REPLACE
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
                    INSERT OR REPLACE INTO stock_data 
                    (ticker, date, open, high, low, close, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(row['ticker']), 
                    date_str, 
                    clean_value(row.get('open')), 
                    clean_value(row.get('high')), 
                    clean_value(row.get('low')), 
                    clean_value(row.get('close')), 
                    clean_value(row.get('adj_close')), 
                    clean_value(row.get('volume'))
                ))
            conn.commit()
    
    def get_stock_data(self, tickers, start_date=None, end_date=None):
        """
        Retrieve stock data from the database.
        
        Args:
            tickers (list): List of stock ticker symbols
            start_date (str, optional): Start date for data retrieval
            end_date (str, optional): End date for data retrieval
            
        Returns:
            pd.DataFrame: DataFrame with stock data
        """
        with sqlite3.connect(self.db_path) as conn:
            # Build query
            placeholders = ','.join(['?' for _ in tickers])
            query = f'''
                SELECT ticker, date, open, high, low, close, adj_close, volume
                FROM stock_data
                WHERE ticker IN ({placeholders})
            '''
            
            params = list(tickers)
            
            if start_date:
                query += ' AND date >= ?'
                params.append(start_date)
            
            if end_date:
                query += ' AND date <= ?'
                params.append(end_date)
            
            query += ' ORDER BY ticker, date'
            
            # Execute query and return DataFrame
            df = pd.read_sql_query(query, conn, params=params)
            # Handle timezone-aware date parsing
            df['date'] = pd.to_datetime(df['date'], format='mixed', utc=True).dt.tz_convert(None)
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
    
    def delete_stock_data(self, ticker):
        """
        Delete all data for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol to delete
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM stock_data WHERE ticker = ?', (ticker,))
            conn.commit()
    
    def get_data_date_range(self, ticker):
        """
        Get the date range of available data for a ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            tuple: (start_date, end_date) or (None, None) if no data
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT MIN(date), MAX(date) 
                FROM stock_data 
                WHERE ticker = ?
            ''', (ticker,))
            
            result = cursor.fetchone()
            return result if result and result[0] else (None, None)
    
    def check_data_freshness(self, ticker, start_date, end_date):
        """
        Check if we have fresh data for a ticker in the requested date range.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Requested start date
            end_date (str): Requested end date
            
        Returns:
            dict: Information about data availability and freshness
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check what data we have
            cursor.execute('''
                SELECT MIN(date), MAX(date), COUNT(*) 
                FROM stock_data 
                WHERE ticker = ? AND date BETWEEN ? AND ?
            ''', (ticker, start_date, end_date))
            
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