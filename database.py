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
                cursor.execute('''
                    INSERT OR REPLACE INTO stock_data 
                    (ticker, date, open, high, low, close, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['ticker'], 
                    row['date'], 
                    row.get('open'), 
                    row.get('high'), 
                    row.get('low'), 
                    row.get('close'), 
                    row.get('adj_close'), 
                    row.get('volume')
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
            df['date'] = pd.to_datetime(df['date'])
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