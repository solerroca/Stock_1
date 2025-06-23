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
    
    def fetch_stock_data(self, ticker, start_date, end_date, interval='1d'):
        """
        Fetch historical stock data for a single ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval ('1d' for daily, '1wk' for weekly)
            
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
                    data = stock.history(
                        start=start_date, 
                        end=end_date, 
                        interval=interval,
                        auto_adjust=True, 
                        prepost=True
                    )
                except Exception as e:
                    # Fallback to period-based fetching with better period selection
                    
                    # Calculate days difference to choose appropriate period
                    from datetime import datetime
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    days_diff = (end_dt - start_dt).days
                    
                    # Choose period based on requested timeframe
                    if days_diff <= 60:
                        period = "3mo"
                    elif days_diff <= 365:
                        period = "1y"
                    elif days_diff <= 730:
                        period = "2y"
                    elif days_diff <= 1825:
                        period = "5y"
                    else:
                        period = "max"
                    
                    try:
                        data = stock.history(
                            period=period, 
                            interval=interval,
                            auto_adjust=True
                        )
                    except Exception as e2:
                        data = pd.DataFrame()  # Empty dataframe
                
                if data.empty:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        pass
                        return None
                
                # Clean the data
                data = data.dropna()
                
                if len(data) < 10:  # Need at least 10 data points
                    pass
                    return None
                
                # --- FIX: Standardize column names to lowercase ---
                data.columns = [col.lower() for col in data.columns]
                
                # Normalize timezone to avoid comparison issues
                data.index = self._normalize_timezone(data.index)
                
                return data
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    pass
                    return None
        
        return None
    
    def fetch_multiple_stocks(self, tickers, start_date, end_date, interval='1d', progress_callback=None):
        """
        Fetch historical stock data for multiple tickers.
        
        Args:
            tickers (list): List of stock ticker symbols
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval ('1d' for daily, '1wk' for weekly)
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
            
            data = self.fetch_stock_data(ticker, start_date, end_date, interval)
            
            if data is not None:
                stock_data[ticker] = data
                successful_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
        
        # Return stock data with status information (don't display messages here)
        return stock_data, successful_tickers, failed_tickers
    
    def fetch_stocks_with_timeframe(self, tickers, timeframe_config, db, progress_callback=None):
        """
        Fetch stock data from the API and store it in the database.
        The decision to use the cache should be made by the caller.
        
        Args:
            tickers (list): List of stock ticker symbols
            timeframe_config (dict): Timeframe configuration with 'days' and 'label'
            db (StockDatabase): Database instance for caching
            progress_callback (function): Optional callback for progress updates
            
        Returns:
            tuple: (stock_data_dict, cache_info, successful_tickers, failed_tickers)
        """
        from datetime import datetime, timedelta, date
        
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=timeframe_config['days'])
        
        # Determine storage strategy based on timeframe
        is_long_term = timeframe_config['days'] >= 730  # 2Y and above use weekly
        frequency = 'weekly' if is_long_term else 'daily'
        api_interval = '1wk' if is_long_term else '1d'
        
        stock_data = {}
        cache_info = {}
        successful_tickers = []
        failed_tickers = []
        
        for i, ticker in enumerate(tickers):
            ticker = ticker.upper().strip()
            
            if progress_callback:
                progress_callback(i, len(tickers), f"Processing {ticker}...")

            # This function now ONLY fetches from the API. Caching is handled in app.py.
            if progress_callback:
                progress_callback(i, len(tickers), f"Fetching {ticker} from API...")
            
            time.sleep(0.1)  # Be respectful to API
            
            api_data = self.fetch_stock_data(
                ticker, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d'),
                interval=api_interval
            )
            
            if api_data is not None and not api_data.empty:
                try:
                    # Store the fetched data in the database
                    db.insert_stock_data(ticker, api_data, frequency=frequency)
                    stock_data[ticker] = api_data
                    successful_tickers.append(ticker)
                    cache_info[ticker] = "fetched"
                    
                except Exception as e:
                    # If storage fails, still use the data but mark as failed cache
                    stock_data[ticker] = api_data
                    successful_tickers.append(ticker)
                    cache_info[ticker] = "fetched_no_cache"
            else:
                failed_tickers.append(ticker)
                cache_info[ticker] = "failed"
        
        return stock_data, cache_info, successful_tickers, failed_tickers
    
    def get_stock_info(self, ticker, db=None):
        """
        Get basic information about a stock with caching.
        
        Args:
            ticker (str): Stock ticker symbol
            db (StockDatabase, optional): Database instance for caching
            
        Returns:
            dict: Dictionary with stock information
        """
        # Try to get cached info first
        if db and db.is_stock_info_fresh(ticker, max_age_days=30):
            cached_info = db.get_cached_stock_info(ticker)
            if cached_info:
                return cached_info
        
        # If no cache or stale data, fetch from API
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            stock_info = {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A')
            }
            
            # Cache the fresh data
            if db:
                try:
                    db.store_stock_info(ticker, stock_info)
                except Exception as e:
                    # If caching fails, still return the data
                    pass
            
            return stock_info
            
        except Exception:
            # If API fails, try to get cached data even if stale
            if db:
                cached_info = db.get_cached_stock_info(ticker)
                if cached_info:
                    return cached_info
            
            # Final fallback
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
        if data.empty or 'close' not in data.columns:
            return {}
        
        try:
            # Calculate returns
            returns = data['close'].pct_change().dropna()
            
            # Calculate metrics
            total_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
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
                'start_price': round(data['close'].iloc[0], 2),
                'end_price': round(data['close'].iloc[-1], 2),
                'trading_days': len(data)
            }
            
        except Exception as e:
            st.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def normalize_prices(self, stock_data_dict, base_value=100):
        """
        Calculate percentage change from starting price for comparison.
        All stocks will start at 0% from the earliest common date.
        
        Args:
            stock_data_dict (dict): Dictionary with ticker as key and DataFrame as value
            base_value (int): Not used for percentage calculation (kept for compatibility)
            
        Returns:
            pd.DataFrame: DataFrame with percentage changes
        """
        if not stock_data_dict:
            return pd.DataFrame()
        
        # Find the earliest common start date across all stocks
        earliest_dates = []
        for ticker, data in stock_data_dict.items():
            if not data.empty:
                earliest_dates.append(data.index.min())
        
        if not earliest_dates:
            return pd.DataFrame()
        
        # Use the latest of the earliest dates as the common start date
        # This ensures all stocks have data from this date forward
        common_start_date = max(earliest_dates)
        
        all_series = []
        
        for ticker, data in stock_data_dict.items():
            if not data.empty and 'close' in data.columns:
                # Filter data to start from common start date
                filtered_data = data[data.index >= common_start_date]['close'].dropna()
                
                if len(filtered_data) > 0:
                    # Calculate percentage change from the price at common start date
                    first_price = filtered_data.iloc[0]
                    if first_price > 0:  # Ensure we don't divide by zero
                        percentage_change = ((filtered_data / first_price) - 1) * 100
                        percentage_change.name = ticker # Set name for concat
                        all_series.append(percentage_change)
        
        # Combine all series at once, which handles different index lengths correctly
        if not all_series:
            return pd.DataFrame()
            
        percentage_data = pd.concat(all_series, axis=1)
        
        # The ffill call below created the "blocky" chart style.
        # By removing it, Plotly will correctly draw lines between
        # real data points (e.g., Friday to Monday) instead of filling weekends.
        # percentage_data = percentage_data.ffill()
        
        # A single dropna at the end cleans up any remaining empty rows
        percentage_data = percentage_data.dropna(how='all')
        
        return percentage_data
    
    def _normalize_timezone(self, datetime_index):
        """
        Normalize datetime index to be timezone-naive for consistent handling.
        
        Args:
            datetime_index: pandas DatetimeIndex that may be timezone-aware or naive
            
        Returns:
            pandas DatetimeIndex: timezone-naive DatetimeIndex
        """
        # Handle non-DatetimeIndex cases
        if not isinstance(datetime_index, pd.DatetimeIndex):
            try:
                datetime_index = pd.DatetimeIndex(datetime_index)
            except:
                return datetime_index
        
        # Remove any NaT values that might cause issues
        if datetime_index.isna().any():
            datetime_index = datetime_index.dropna()
        
        if hasattr(datetime_index, 'tz') and datetime_index.tz is not None:
            # Convert timezone-aware to timezone-naive
            try:
                return datetime_index.tz_convert(None)
            except:
                # If conversion fails, try localize to None
                try:
                    return datetime_index.tz_localize(None)
                except:
                    # If all else fails, create new timezone-naive index
                    try:
                        return pd.DatetimeIndex(datetime_index.values, tz=None)
                    except:
                        # Final fallback: recreate from string representation
                        return pd.DatetimeIndex([pd.Timestamp(ts).tz_localize(None) if pd.notna(ts) else pd.NaT for ts in datetime_index])
        
        # Already timezone-naive
        return datetime_index
    
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