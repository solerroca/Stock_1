"""
Stock Portfolio Tracker - Main Streamlit Application

A web application for tracking and visualizing stock performance.
This application allows users to:
- Enter up to 10 stock tickers
- Fetch historical stock data
- Store data in SQLite database
- Display interactive charts and performance metrics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
import time

# Import our custom modules
from database import StockDatabase
from stock_data import StockDataFetcher


def normalize_timezone_index(datetime_index):
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


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = {}
    if 'db' not in st.session_state:
        st.session_state.db = StockDatabase()
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = StockDataFetcher()
    if 'ticker_input' not in st.session_state:
        st.session_state.ticker_input = ""
    if 'last_cleanup_check' not in st.session_state:
        st.session_state.last_cleanup_check = None
    if 'selected_timeframe' not in st.session_state:
        st.session_state.selected_timeframe = '1Y'
    if 'current_tickers' not in st.session_state:
        st.session_state.current_tickers = []


def create_sidebar():
    """Create the application sidebar with controls."""
    st.sidebar.title("🎯 Portfolio Controls")
    
    # Stock ticker input
    st.sidebar.subheader("Stock Tickers")
    st.sidebar.markdown("Enter up to 10 stock symbols (e.g., AAPL, GOOGL, MSFT)")
    
    # Text area for multiple tickers
    tickers_input = st.sidebar.text_area(
        "Stock Symbols (comma-separated):",
        value=st.session_state.ticker_input,
        placeholder="AAPL, GOOGL, MSFT, AMZN, TSLA",
        height=100,
        key="ticker_text_area"
    )
    
    # Popular tickers suggestions
    popular_tickers = st.session_state.data_fetcher.get_popular_tickers()
    st.sidebar.markdown("**Popular Stocks:**")
    st.sidebar.markdown(", ".join(popular_tickers[:10]))
    
    # Update session state with current input
    st.session_state.ticker_input = tickers_input
    
    # Parse tickers
    tickers = []
    if tickers_input:
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
        tickers = tickers[:10]  # Limit to 10 tickers
    
    # Fetch data button
    fetch_button = st.sidebar.button(
        "📊 Fetch Stock Data", 
        type="primary",
        use_container_width=True
    )
    
    # Database info (read-only)
    available_tickers = st.session_state.db.get_available_tickers()
    if available_tickers:
        st.sidebar.markdown(f"**📊 Cached Tickers:** {len(available_tickers)}")
        st.sidebar.caption("Database auto-manages storage size")
    
    return tickers, fetch_button


def create_timeframe_buttons():
    """Create timeframe selection buttons."""
    st.subheader("📅 Select Time Period")
    
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    timeframes = {
        '1M': {'label': '1 Month', 'days': 30, 'col': col1},
        '3M': {'label': '3 Months', 'days': 90, 'col': col2},
        '6M': {'label': '6 Months', 'days': 180, 'col': col3},
        '1Y': {'label': '1 Year', 'days': 365, 'col': col4},
        '2Y': {'label': '2 Years', 'days': 730, 'col': col5},
        '3Y': {'label': '3 Years', 'days': 1095, 'col': col6},
        '5Y': {'label': '5 Years', 'days': 1825, 'col': col7}
    }
    
    for timeframe_key, config in timeframes.items():
        with config['col']:
            button_type = "primary" if st.session_state.selected_timeframe == timeframe_key else "secondary"
            if st.button(
                config['label'], 
                key=f"btn_{timeframe_key}",
                type=button_type,
                use_container_width=True
            ):
                st.session_state.selected_timeframe = timeframe_key
                # Trigger data refresh if we have tickers
                if st.session_state.current_tickers:
                    st.rerun()
    
    return timeframes[st.session_state.selected_timeframe]


def create_progress_callback():
    """Create a progress callback for data fetching."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(current, total, message):
        progress = (current + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"{message} ({current + 1}/{total})")
    
    return update_progress, progress_bar, status_text


def display_stock_info(tickers):
    """Display basic information about selected stocks."""
    if not tickers:
        return
    
    st.subheader("📈 Stock Information")
    
    info_data = []
    for ticker in tickers:
        info = st.session_state.data_fetcher.get_stock_info(ticker)
        info_data.append({
            'Ticker': ticker,
            'Company': info['name'],
            'Sector': info['sector'],
            'P/E Ratio': info['pe_ratio'] if info['pe_ratio'] != 'N/A' else 'N/A'
        })
    
    if info_data:
        info_df = pd.DataFrame(info_data)
        st.dataframe(info_df, use_container_width=True)


def filter_data_by_timeframe(stock_data_dict, days):
    """Filter stock data to show only the last N days from the available data."""
    from datetime import timedelta
    import pandas as pd
    
    if not stock_data_dict:
        return {}
    
    # Find the common date range across all stocks
    all_start_dates = []
    all_end_dates = []
    
    for ticker, data in stock_data_dict.items():
        if not data.empty:
            # Normalize timezone information - convert to timezone-naive
            index = data.index
            index = normalize_timezone_index(index)
            
            all_start_dates.append(index.min())
            all_end_dates.append(index.max())
    
    if not all_start_dates:
        return {}
    
    # Use the latest start date and earliest end date to ensure all stocks have data
    common_start = max(all_start_dates)
    common_end = min(all_end_dates)
    
    # Calculate cutoff date from the common end date
    cutoff_date = common_end - timedelta(days=days)
    
    # Use the later of cutoff_date or common_start to ensure we don't go beyond available data
    final_cutoff = max(cutoff_date, common_start)
    
    filtered_data = {}
    for ticker, data in stock_data_dict.items():
        if not data.empty:
            # Normalize the data index timezone for comparison
            data_copy = data.copy()
            data_copy.index = normalize_timezone_index(data_copy.index)
            
            # Filter data to the common timeframe
            filtered_df = data_copy[(data_copy.index >= final_cutoff) & (data_copy.index <= common_end)]
            if not filtered_df.empty:
                filtered_data[ticker] = filtered_df
    
    return filtered_data


def create_price_chart(stock_data_dict):
    """Create an interactive price chart."""
    if not stock_data_dict:
        return
    
    st.subheader("📊 Stock Performance Comparison")
    
    # Use the selected timeframe from the main timeframe buttons
    if 'selected_timeframe' not in st.session_state:
        st.session_state.selected_timeframe = '1Y'
    
    time_periods = {
        "1M": 30,
        "3M": 90, 
        "6M": 180,
        "1Y": 365,
        "2Y": 730,
        "3Y": 1095,
        "5Y": 1825
    }
    
    # Filter data based on selected timeframe
    filtered_stock_data = filter_data_by_timeframe(stock_data_dict, time_periods[st.session_state.selected_timeframe])
    
    # Check if we're showing all available data (timeframe exceeds available data)
    show_data_limit_warning = False
    if filtered_stock_data:
        for ticker, data in filtered_stock_data.items():
            original_data = stock_data_dict[ticker]
            if len(data) == len(original_data):
                show_data_limit_warning = True
                break
    
    # Create percentage change data for comparison
    percentage_data = st.session_state.data_fetcher.normalize_prices(filtered_stock_data)
    
    # Debug: Show what data we have
    with st.expander("🔍 Debug: Chart Data Info", expanded=False):
        st.write("**Filtered Stock Data:**")
        for ticker, data in filtered_stock_data.items():
            st.write(f"- {ticker}: {len(data)} rows, dates from {data.index.min()} to {data.index.max()}")
        
        st.write("**Percentage Data:**")
        st.write(f"- Shape: {percentage_data.shape}")
        st.write(f"- Columns: {list(percentage_data.columns)}")
        if not percentage_data.empty:
            st.write("- Sample data:")
            st.dataframe(percentage_data.head())
    
    if percentage_data.empty:
        st.warning("No data available for charting in the selected timeframe")
        return
    
    # Show warning if timeframe exceeds available data
    if show_data_limit_warning:
        timeframe_labels = {
            "1M": "1 Month",
            "3M": "3 Months", 
            "6M": "6 Months",
            "1Y": "1 Year",
            "2Y": "2 Years",
            "3Y": "3 Years",
            "5Y": "5 Years"
        }
        st.info(f"ℹ️ Showing all available data (less than {timeframe_labels[st.session_state.selected_timeframe]} available)")
    
    # Create plotly figure
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, ticker in enumerate(percentage_data.columns):
        # Get the data for this ticker and drop NaN values
        ticker_data = percentage_data[ticker].dropna()
        
        if len(ticker_data) > 0:  # Only add trace if we have valid data
            fig.add_trace(go.Scatter(
                x=ticker_data.index,
                y=ticker_data.values,
                mode='lines',
                name=ticker,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>{ticker}</b><br>Date: %{{x}}<br>Change: %{{y:.1f}}%<extra></extra>'
            ))
    
    # Add a horizontal line at 0% for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Get timeframe label for title
    timeframe_labels = {
        "1M": "1 Month",
        "3M": "3 Months", 
        "6M": "6 Months",
        "1Y": "1 Year",
        "2Y": "2 Years",
        "3Y": "3 Years",
        "5Y": "5 Years"
    }
    
    fig.update_layout(
        title=f"Stock Performance Comparison - {timeframe_labels[st.session_state.selected_timeframe]} (% Change from Start)",
        xaxis_title="Date",
        yaxis_title="Percentage Change (%)",
        hovermode='x unified',
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_performance_table(stock_data_dict):
    """Create a comprehensive performance metrics and stock information table."""
    if not stock_data_dict:
        return
    
    st.subheader("📊 Performance Metrics & Stock Information")
    
    metrics_data = []
    for ticker, data in stock_data_dict.items():
        # Get performance metrics
        metrics = st.session_state.data_fetcher.calculate_performance_metrics(data)
        # Get stock information
        stock_info = st.session_state.data_fetcher.get_stock_info(ticker)
        
        if metrics:
            metrics_data.append({
                'Ticker': ticker,
                'Company': stock_info['name'],
                'Sector': stock_info['sector'],
                'P/E Ratio': stock_info['pe_ratio'] if stock_info['pe_ratio'] != 'N/A' else 'N/A',
                'Total Return (%)': metrics['total_return'],
                'Volatility (%)': metrics['volatility'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown (%)': metrics['max_drawdown'],
                'Start Price ($)': metrics['start_price'],
                'End Price ($)': metrics['end_price'],
                'Trading Days': metrics['trading_days']
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        
        # Style the dataframe
        def color_performance(val):
            if isinstance(val, (int, float)):
                color = 'color: green' if val > 0 else 'color: red' if val < 0 else 'color: black'
                return color
            return ''
        
        styled_df = metrics_df.style.map(
            color_performance, 
            subset=['Total Return (%)', 'Max Drawdown (%)']
        )
        
        st.dataframe(styled_df, use_container_width=True)


def create_volume_chart(stock_data_dict):
    """Create a volume chart."""
    if not stock_data_dict or len(stock_data_dict) == 0:
        return
    
    st.subheader("📊 Trading Volume")
    
    # Use the same timeframe as the price chart
    if 'selected_timeframe' in st.session_state:
        time_periods = {
            "1M": 30,
            "3M": 90,
            "6M": 180,
            "1Y": 365,
            "2Y": 730,
            "3Y": 1095,
            "5Y": 1825
        }
        filtered_stock_data = filter_data_by_timeframe(stock_data_dict, time_periods[st.session_state.selected_timeframe])
    else:
        filtered_stock_data = stock_data_dict
    
    # Select ticker for volume chart
    selected_ticker = st.selectbox(
        "Select ticker for volume chart:",
        list(filtered_stock_data.keys()) if filtered_stock_data else []
    )
    
    if selected_ticker and selected_ticker in filtered_stock_data:
        data = filtered_stock_data[selected_ticker]
        
        # Debug volume data
        st.write(f"**Volume Debug for {selected_ticker}:**")
        st.write(f"- Data shape: {data.shape}")
        st.write(f"- Columns: {list(data.columns)}")
        st.write(f"- Has Volume column: {'Volume' in data.columns}")
        if 'Volume' in data.columns:
            st.write(f"- Volume range: {data['Volume'].min()} to {data['Volume'].max()}")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='lightblue',
            hovertemplate='<b>Volume</b><br>Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
        ))
        
        # Get timeframe label for title
        timeframe_labels = {
            "1M": "1 Month",
            "3M": "3 Months", 
            "6M": "6 Months",
            "1Y": "1 Year",
            "2Y": "2 Years",
            "3Y": "3 Years",
            "5Y": "5 Years"
        }
        
        timeframe_label = ""
        if 'selected_timeframe' in st.session_state:
            timeframe_label = f" - {timeframe_labels[st.session_state.selected_timeframe]}"
        
        fig.update_layout(
            title=f"Trading Volume - {selected_ticker}{timeframe_label}",
            xaxis_title="Date",
            yaxis_title="Volume",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def perform_automatic_cleanup():
    """Perform automatic database cleanup if needed."""
    from datetime import datetime, timedelta
    
    # Only check cleanup every hour to avoid frequent checks
    now = datetime.now()
    if (st.session_state.last_cleanup_check is None or 
        now - st.session_state.last_cleanup_check > timedelta(hours=1)):
        
        st.session_state.last_cleanup_check = now
        
        # Perform cleanup if needed
        cleanup_result = st.session_state.db.cleanup_old_data(max_size_mb=50, max_age_days=180)
        
        if cleanup_result and cleanup_result.get('cleanup_performed'):
            # Store cleanup message for display
            st.session_state.cleanup_message = (
                f"🧹 Database cleanup: Removed {cleanup_result['total_deleted']} old records "
                f"to optimize storage"
            )
            st.session_state.show_cleanup_message = True


def main():
    """Main application function."""
    # Configure page
    st.set_page_config(
        page_title="Stock Portfolio Tracker",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Perform automatic cleanup check
    perform_automatic_cleanup()
    
    # Main title
    st.title("📈 Stock Portfolio Tracker")
    st.markdown("Track and analyze your favorite stocks with interactive charts and performance metrics.")
    
    # Create sidebar
    tickers, fetch_button = create_sidebar()
    
    # Main content area
    if fetch_button and tickers:
        if len(tickers) > 10:
            st.error("Please enter no more than 10 stock tickers.")
        else:
            # Store current tickers for timeframe changes
            st.session_state.current_tickers = tickers
            
            # Get timeframe configuration
            timeframe_config = create_timeframe_buttons()
            
            # Create progress tracking
            progress_callback, progress_bar, status_text = create_progress_callback()
            
            # Use smart data fetching with timeframe support
            result = st.session_state.data_fetcher.fetch_stocks_with_timeframe(
                tickers,
                timeframe_config,
                st.session_state.db,
                progress_callback
            )
            
            # Unpack the result
            stock_data, cache_info, successful_tickers, failed_tickers = result
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Store results for display
            if stock_data:
                st.session_state.stock_data = stock_data
                st.session_state.success_status = f"Successfully loaded data for {len(successful_tickers)} stocks"
                st.session_state.show_success_status = True
                
                # Create cache analysis message
                cached_count = sum(1 for status in cache_info.values() if status == "cached")
                fetched_count = sum(1 for status in cache_info.values() if status in ["fetched", "fetched_no_cache"])
                
                cache_analysis = []
                if cached_count > 0:
                    frequency_type = "weekly" if timeframe_config['days'] >= 730 else "daily"
                    cache_analysis.append(f"📊 {cached_count} stocks loaded from cache ({frequency_type})")
                if fetched_count > 0:
                    frequency_type = "weekly" if timeframe_config['days'] >= 730 else "daily"
                    cache_analysis.append(f"🌐 {fetched_count} stocks fetched from Yahoo Finance ({frequency_type})")
                
                if cache_analysis:
                    st.session_state.cache_analysis = " | ".join(cache_analysis)
                    st.session_state.show_cache_analysis = True
            
            if failed_tickers:
                st.session_state.fetch_failed_status = f"Failed to load data for: {', '.join(failed_tickers)}"
                st.session_state.show_fetch_failed_status = True
    
    # Handle timeframe changes for existing data
    elif st.session_state.current_tickers and not fetch_button:
        # Show timeframe buttons
        timeframe_config = create_timeframe_buttons()
        
        # If timeframe changed, reload data
        if st.session_state.current_tickers:
            # Create progress tracking
            progress_callback, progress_bar, status_text = create_progress_callback()
            
            # Use smart data fetching with new timeframe
            result = st.session_state.data_fetcher.fetch_stocks_with_timeframe(
                st.session_state.current_tickers,
                timeframe_config,
                st.session_state.db,
                progress_callback
            )
            
            # Unpack the result
            stock_data, cache_info, successful_tickers, failed_tickers = result
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Update session state
            if stock_data:
                st.session_state.stock_data = stock_data
                
                # Create cache analysis message
                cached_count = sum(1 for status in cache_info.values() if status == "cached")
                fetched_count = sum(1 for status in cache_info.values() if status in ["fetched", "fetched_no_cache"])
                
                cache_analysis = []
                if cached_count > 0:
                    frequency_type = "weekly" if timeframe_config['days'] >= 730 else "daily"
                    cache_analysis.append(f"📊 {cached_count} stocks loaded from cache ({frequency_type})")
                if fetched_count > 0:
                    frequency_type = "weekly" if timeframe_config['days'] >= 730 else "daily"
                    cache_analysis.append(f"🌐 {fetched_count} stocks fetched from Yahoo Finance ({frequency_type})")
                
                if cache_analysis:
                    st.session_state.cache_analysis = " | ".join(cache_analysis)
                    st.session_state.show_cache_analysis = True
    
    # Display data if available
    if st.session_state.stock_data:
        # Show timeframe buttons if we have data but no tickers input
        if not st.session_state.current_tickers:
            create_timeframe_buttons()
        
        # Main price chart (full width for better visibility)
        create_price_chart(st.session_state.stock_data)
        
        # Performance metrics table with stock information (full width between charts)
        create_performance_table(st.session_state.stock_data)
        
        # Volume chart (full width at bottom)
        create_volume_chart(st.session_state.stock_data)
        
        # Display status messages at the bottom
        if hasattr(st.session_state, 'show_success_status') and st.session_state.show_success_status:
            st.success(st.session_state.success_status)
            st.session_state.show_success_status = False
        
        if hasattr(st.session_state, 'show_fetch_failed_status') and st.session_state.show_fetch_failed_status:
            st.warning(st.session_state.fetch_failed_status)
            st.session_state.show_fetch_failed_status = False
        
        if hasattr(st.session_state, 'show_cache_analysis') and st.session_state.show_cache_analysis:
            st.info(st.session_state.cache_analysis)
            st.session_state.show_cache_analysis = False
        
        if hasattr(st.session_state, 'show_cleanup_message') and st.session_state.show_cleanup_message:
            st.info(st.session_state.cleanup_message)
            st.session_state.show_cleanup_message = False
    
    else:
        # Welcome message
        st.markdown("""
        ### Welcome to Stock Portfolio Tracker! 🎯
        
        **How to get started:**
        1. Enter stock ticker symbols in the sidebar (e.g., AAPL, GOOGL, MSFT)
        2. Click "Fetch Stock Data" to load 1-year data by default
        3. Use the timeframe buttons (1Y, 2Y, 3Y, 5Y) to change the time period
        4. View interactive charts and performance metrics
        
        **Features:**
        - 📊 Interactive price comparison charts with percentage changes
        - 📈 Performance metrics and analysis
        - 🗄️ Hybrid storage: daily data for ≤1Y, weekly data for 2Y+ periods
        - ⚡ Optimized performance with direct storage (no resampling)
        - 📱 Responsive design for mobile and desktop
        
        **Tips:**
        - You can track up to 10 stocks at once
        - Data is cached locally for fast access and minimal API usage
        - Short periods (≤1Y) use daily data, longer periods (2Y+) use weekly data
        - Database automatically manages storage size and cleanup
        """)
        
        # Show popular tickers as clickable buttons
        popular_tickers = st.session_state.data_fetcher.get_popular_tickers()
        st.markdown("**💡 Popular Stocks - Click to add:**")
        
        # Create columns for popular ticker buttons
        cols = st.columns(5)
        for i, ticker in enumerate(popular_tickers[:10]):
            with cols[i % 5]:
                if st.button(ticker, key=f"popular_{ticker}"):
                    # Add ticker to input if not already there
                    current_input = st.session_state.ticker_input.strip()
                    if current_input:
                        # Check if ticker is already in the input
                        current_tickers = [t.strip().upper() for t in current_input.split(',')]
                        if ticker not in current_tickers:
                            st.session_state.ticker_input = current_input + f", {ticker}"
                    else:
                        st.session_state.ticker_input = ticker
                    st.rerun()


if __name__ == "__main__":
    main() 