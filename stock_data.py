"""
Stock data module for fetching and processing stock information.

This module handles:
- Fetching stock data from Yahoo Finance
- Data validation and cleaning
- Calculating performance metrics
- Error handling for API requests
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import streamlit as st


class StockDataFetcher:
    """Class to handle stock data fetching and processing."""
    
    def __init__(self):
        """Initialize the StockDataFetcher."""
        pass
    
    def validate_ticker(self, ticker):
        """
        Validate if a ticker symbol exists and has data.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            stock = yf.Ticker(ticker)
            # Try to get recent data to validate ticker
            hist = stock.history(period="5d")
            return not hist.empty
        except Exception:
            return False
    
    def fetch_stock_data(self, ticker, start_date, end_date):
        """
        Fetch historical stock data for a single ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: DataFrame with stock data, or None if error
        """
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Create ticker object with session for better reliability
                stock = yf.Ticker(ticker)
                
                # Fetch data with period as fallback if dates don't work
                try:
                    data = stock.history(start=start_date, end=end_date, auto_adjust=True, prepost=True)
                except:
                    # Fallback to period-based fetching
                    data = stock.history(period="1y", auto_adjust=True)
                
                if data.empty:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        st.warning(f"No data available for ticker: {ticker}")
                        return None
                
                # Clean the data
                data = data.dropna()
                
                if len(data) < 10:  # Need at least 10 data points
                    st.warning(f"Insufficient data for ticker: {ticker}")
                    return None
                
                return data
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    st.error(f"Error fetching data for {ticker}: {str(e)}")
                    return None
        
        return None
    
    def fetch_multiple_stocks(self, tickers, start_date, end_date, progress_callback=None):
        """
        Fetch historical stock data for multiple tickers.
        
        Args:
            tickers (list): List of stock ticker symbols
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            progress_callback (function): Optional callback for progress updates
            
        Returns:
            dict: Dictionary with ticker as key and DataFrame as value
        """
        stock_data = {}
        successful_tickers = []
        failed_tickers = []
        
        for i, ticker in enumerate(tickers):
            ticker = ticker.upper().strip()
            
            if progress_callback:
                progress_callback(i, len(tickers), f"Fetching {ticker}...")
            
            # Add a small delay to be respectful to the API
            time.sleep(0.1)
            
            data = self.fetch_stock_data(ticker, start_date, end_date)
            
            if data is not None:
                stock_data[ticker] = data
                successful_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
        
        # Show summary
        if successful_tickers:
            st.success(f"Successfully fetched data for: {', '.join(successful_tickers)}")
        
        if failed_tickers:
            st.warning(f"Failed to fetch data for: {', '.join(failed_tickers)}")
        
        return stock_data
    
    def get_stock_info(self, ticker):
        """
        Get basic information about a stock.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Dictionary with stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A')
            }
        except Exception:
            return {
                'name': ticker,
                'sector': 'N/A',
                'industry': 'N/A', 
                'market_cap': 'N/A',
                'pe_ratio': 'N/A',
                'dividend_yield': 'N/A'
            }
    
    def calculate_performance_metrics(self, data):
        """
        Calculate performance metrics for stock data.
        
        Args:
            data (pd.DataFrame): Stock data DataFrame
            
        Returns:
            dict: Dictionary with performance metrics
        """
        if data.empty:
            return {}
        
        try:
            # Calculate returns
            returns = data['Close'].pct_change().dropna()
            
            # Calculate metrics
            total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            avg_daily_return = returns.mean() * 100
            volatility = returns.std() * 100
            
            # Calculate Sharpe ratio (assuming risk-free rate of 2%)
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            excess_returns = returns - risk_free_rate
            sharpe_ratio = excess_returns.mean() / returns.std() if returns.std() != 0 else 0
            
            # Calculate max drawdown
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            return {
                'total_return': round(total_return, 2),
                'avg_daily_return': round(avg_daily_return, 4),
                'volatility': round(volatility, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'max_drawdown': round(max_drawdown, 2),
                'start_price': round(data['Close'].iloc[0], 2),
                'end_price': round(data['Close'].iloc[-1], 2),
                'trading_days': len(data)
            }
            
        except Exception as e:
            st.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def normalize_prices(self, stock_data_dict, base_value=100):
        """
        Calculate percentage change from starting price for comparison.
        
        Args:
            stock_data_dict (dict): Dictionary with ticker as key and DataFrame as value
            base_value (int): Not used for percentage calculation (kept for compatibility)
            
        Returns:
            pd.DataFrame: DataFrame with percentage changes
        """
        percentage_data = pd.DataFrame()
        
        for ticker, data in stock_data_dict.items():
            if not data.empty:
                # Calculate percentage change from first price
                percentage_change = ((data['Close'] / data['Close'].iloc[0]) - 1) * 100
                percentage_data[ticker] = percentage_change
        
        return percentage_data
    
    @staticmethod
    def get_popular_tickers():
        """
        Get a list of popular stock tickers for suggestions.
        
        Returns:
            list: List of popular ticker symbols
        """
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
            'META', 'NVDA', 'BRK-B', 'JNJ', 'V',
            'WMT', 'PG', 'UNH', 'HD', 'MA',
            'DIS', 'PYPL', 'ADBE', 'NFLX', 'CRM'
        ] 