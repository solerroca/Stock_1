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
    if 'db' not in st.session_state:
        st.session_state.db = StockDatabase()
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = StockDataFetcher()
        
    # App logic state
    if 'daily_data' not in st.session_state:
        st.session_state.daily_data = {}
    if 'weekly_data' not in st.session_state:
        st.session_state.weekly_data = {}
    if 'ticker_input' not in st.session_state:
        st.session_state.ticker_input = "AAPL, MSFT, AMZN"
    if 'current_tickers' not in st.session_state:
        st.session_state.current_tickers = []
    if 'selected_timeframe' not in st.session_state:
        st.session_state.selected_timeframe = '1Y'
    if 'last_fetched_weekly_timeframe' not in st.session_state:
        st.session_state.last_fetched_weekly_timeframe = None
        
    # Control flags
    if 'first_load' not in st.session_state:
        st.session_state.first_load = True
    if 'show_instructions' not in st.session_state:
        st.session_state.show_instructions = False
        
    # Status messages
    if 'last_cleanup_check' not in st.session_state:
        st.session_state.last_cleanup_check = None


def show_how_to_use_content():
    """Display instructions content for the app."""
    st.markdown("""
    ## 🚀 Getting Started
    
    ### 1. **Adding Stocks**
    - **Manual Entry**: Type stock symbols in the text area (e.g., AAPL, GOOGL, MSFT)
    - **Quick Add**: Click buttons from Popular Stocks, ETFs, or Crypto sections
    - **Limit**: You can track up to 10 stocks at once
    
    ### 2. **Fetching Data**
    - Click **"📊 Fetch Stock Data"** to load current data
    - The app automatically loads AAPL, MSFT, AMZN on first visit
    - Data is cached for fast access and minimal API usage
    
    ### 3. **Analyzing Performance**
    - **Time Periods**: Use buttons (1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y) to change timeframes
    - **Chart Reading**: All stocks start at 0% to show relative performance
    - **Gains vs. Losses**: Lines above 0% indicate gains; below 0% show losses.
    
    ### 4. **Understanding the Data**
    - **Performance Chart**: Shows percentage change over time
    - **Metrics Table**: Displays returns, volatility, Sharpe ratio, and company info
    - **Volume Chart**: Select any stock to view its trading volume
    
    ### 5. **Data Storage**
    - **Smart Storage**: Daily data for ≤1Y, weekly data for 2Y+ periods
    - **Auto-Cleanup**: Database manages size automatically
    - **Cached Data**: Previously fetched data loads instantly
    
    ### 💡 **Pro Tips**
    - Mix stocks, ETFs, and crypto for diversified analysis
    - Use longer timeframes (2Y+) for trend analysis
    - Compare similar companies or sectors for insights
    - Check the Sharpe ratio to evaluate risk-adjusted returns
    
    ### 🔧 **Troubleshooting**
    - If a stock fails to load, check the ticker symbol
    - Some crypto symbols need "-USD" suffix (handled automatically)
    - Data updates daily - cache refreshes as needed
    """)


def create_sidebar():
    """Create the application sidebar with controls."""
    st.sidebar.title("🎯 Portfolio Controls")
    
    if st.sidebar.button("❓ How to Use", use_container_width=True):
        st.session_state.show_instructions = not st.session_state.show_instructions
        st.rerun()
    
    st.sidebar.subheader("Stock Tickers")
    st.sidebar.markdown("Enter up to 10 stock symbols (e.g., AAPL, GOOGL, MSFT)")
    
    tickers_input = st.sidebar.text_area(
        "Stock Symbols (comma-separated):",
        value=st.session_state.ticker_input,
        placeholder="AAPL, GOOGL, MSFT",
        height=60,
        key="ticker_text_area"
    )
    st.session_state.ticker_input = tickers_input
    
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()][:10]
    
    fetch_button = st.sidebar.button("📊 Fetch Stock Data", type="primary", use_container_width=True)
    
    st.sidebar.markdown("**Popular Stocks:**")
    popular_tickers = st.session_state.data_fetcher.get_popular_tickers()[:12]
    cols = st.sidebar.columns(3)
    for i, ticker in enumerate(popular_tickers):
        if cols[i % 3].button(ticker, key=f"pop_stock_{ticker}", use_container_width=True):
            current_tickers = [t.strip().upper() for t in st.session_state.ticker_input.split(',') if t.strip()]
            if ticker not in current_tickers:
                st.session_state.ticker_input += f", {ticker}"
                st.rerun()

    st.sidebar.markdown("**Popular ETFs:**")
    popular_etfs = ['SPY', 'QQQ', 'VTI', 'IWM', 'EFA', 'VEA', 'VWO', 'GLD', 'SLV', 'TLT', 'HYG', 'LQD']
    etf_cols = st.sidebar.columns(3)
    for i, etf in enumerate(popular_etfs):
        if etf_cols[i % 3].button(etf, key=f"pop_etf_{etf}", use_container_width=True):
            current_tickers = [t.strip().upper() for t in st.session_state.ticker_input.split(',') if t.strip()]
            if etf not in current_tickers:
                st.session_state.ticker_input += f", {etf}"
                st.rerun()

    st.sidebar.markdown("**Popular Crypto:**")
    popular_crypto = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD', 'LTC-USD', 'LINK-USD']
    crypto_cols = st.sidebar.columns(3)
    for i, crypto in enumerate(popular_crypto):
        crypto_display = crypto.replace('-USD', '')
        if crypto_cols[i % 3].button(crypto_display, key=f"pop_crypto_{crypto}", use_container_width=True):
            current_tickers = [t.strip().upper() for t in st.session_state.ticker_input.split(',') if t.strip()]
            if crypto not in current_tickers:
                st.session_state.ticker_input += f", {crypto}"
                st.rerun()

    available_tickers = st.session_state.db.get_available_tickers()
    if available_tickers:
        st.sidebar.markdown(f"**📊 Cached Tickers:** {len(available_tickers)}")
        st.sidebar.caption("Database auto-manages storage size")
    
    return tickers, fetch_button


def create_timeframe_buttons():
    """Create and manage timeframe selection buttons."""
    timeframes = {
        '1M': 30, '3M': 90, '6M': 180, '1Y': 365,
        '2Y': 730, '3Y': 1095, '5Y': 1825
    }
    timeframe_labels = {
        '1M': '1 Month', '3M': '3 Months', '6M': '6 Months', '1Y': '1 Year',
        '2Y': '2 Years', '3Y': '3 Years', '5Y': '5 Years'
    }
    
    cols = st.columns(len(timeframes))
    
    # Store the previous timeframe to detect changes
    prev_timeframe = st.session_state.selected_timeframe
    
    for i, (key, label) in enumerate(timeframe_labels.items()):
        button_type = "primary" if st.session_state.selected_timeframe == key else "secondary"
        if cols[i].button(label, key=f"btn_{key}", type=button_type, use_container_width=True):
            st.session_state.selected_timeframe = key
            # Rerun only if a long-term timeframe is selected to trigger a potential fetch
            if prev_timeframe != key and timeframes[key] >= 730:
                st.rerun()
                
    return timeframes, timeframes[st.session_state.selected_timeframe]


def create_progress_callback():
    """Create a progress callback for data fetching."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(current, total, message):
        progress = (current + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"{message} ({current + 1}/{total})")
    
    return update_progress, progress_bar, status_text


def filter_data_by_timeframe(stock_data_dict, days):
    """
    Filter stock data for a specific timeframe from the most recent date available.
    """
    if not stock_data_dict:
        return {}

    from datetime import timedelta
    
    latest_end_date = None
    for data in stock_data_dict.values():
        if not data.empty:
            current_max_date = normalize_timezone_index(data.index).max()
            if latest_end_date is None or current_max_date > latest_end_date:
                latest_end_date = current_max_date
    
    if latest_end_date is None:
        return {}
        
    start_date_cutoff = latest_end_date - timedelta(days=days)

    filtered_data = {}
    for ticker, data in stock_data_dict.items():
        if not data.empty:
            data_copy = data.copy()
            data_copy.index = normalize_timezone_index(data_copy.index)
            mask = data_copy.index >= start_date_cutoff
            filtered_df = data_copy.loc[mask]

            if not filtered_df.empty:
                filtered_data[ticker] = filtered_df
    
    return filtered_data


def create_price_chart(filtered_stock_data):
    """Create an interactive price chart from pre-filtered data."""
    if not filtered_stock_data:
        return
    
    st.subheader("📊 Stock Performance Comparison")
    
    percentage_data = st.session_state.data_fetcher.normalize_prices(filtered_stock_data)
    
    st.info("📈 **Chart Explanation:** This chart shows the percentage change in stock prices from the start of the selected time period. All stocks begin at 0% and show their relative performance over time. A line above 0% indicates gains, while below 0% shows losses.")
    
    if percentage_data.empty:
        st.warning("Could not normalize data for charting in the selected timeframe.")
        return
    
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    
    for i, ticker in enumerate(percentage_data.columns):
        ticker_data = percentage_data[ticker].dropna()
        if not ticker_data.empty:
            fig.add_trace(go.Scatter(
                x=ticker_data.index, y=ticker_data.values, mode='lines', name=ticker,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>{ticker}</b><br>Date: %{{x|%Y-%m-%d}}<br>Change: %{{y:.2f}}%<extra></extra>'
            ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    
    timeframe_labels = {'1M': '1 Month', '3M': '3 Months', '6M': '6 Months', '1Y': '1 Year', '2Y': '2 Years', '3Y': '3 Years', '5Y': '5 Years'}
    title = f"Stock Performance Comparison - {timeframe_labels[st.session_state.selected_timeframe]} (% Change from Start)"
    
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Percentage Change (%)",
        hovermode='x unified', template="plotly_white", height=500, legend_title_text='Ticker'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_performance_table(filtered_stock_data):
    """Create a comprehensive performance metrics table from pre-filtered data."""
    if not filtered_stock_data:
        return
    
    st.subheader("📊 Performance Metrics & Stock Information")
    
    metrics_data = []
    for ticker, data in filtered_stock_data.items():
        metrics = st.session_state.data_fetcher.calculate_performance_metrics(data)
        stock_info = st.session_state.data_fetcher.get_stock_info(ticker, st.session_state.db)
        
        # This check ensures that we only proceed if metrics were successfully calculated
        if metrics:
            pe_ratio = stock_info.get('pe_ratio')
            pe_ratio_display = None
            if pe_ratio and pe_ratio != 'N/A':
                try:
                    pe_ratio_display = float(pe_ratio)
                except (ValueError, TypeError):
                    pe_ratio_display = None # Keep as None if conversion fails
            
            # Build the dictionary with the correct, expected column names
            metrics_data.append({
                'Ticker': ticker,
                'Company': stock_info.get('name', ticker),
                'Sector': stock_info.get('sector', 'N/A'),
                'P/E Ratio': pe_ratio_display,
                'Total Return (%)': metrics.get('total_return'),
                'Volatility (%)': metrics.get('volatility'),
                'Sharpe Ratio': metrics.get('sharpe_ratio'),
                'Max Drawdown (%)': metrics.get('max_drawdown'),
                'Start Price ($)': metrics.get('start_price'),
                'End Price ($)': metrics.get('end_price'),
                'Trading Days': metrics.get('trading_days')
            })
    
    if metrics_data:
        # Define the exact order of columns
        column_order = ['Ticker', 'Company', 'Sector', 'P/E Ratio', 'Total Return (%)', 
                        'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 
                        'Start Price ($)', 'End Price ($)', 'Trading Days']
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Ensure all required columns exist, adding any that are missing as None
        for col in column_order:
            if col not in metrics_df.columns:
                metrics_df[col] = None

        # Reorder the DataFrame safely
        metrics_df = metrics_df[column_order]

        styled_df = metrics_df.style.format({
            'P/E Ratio': '{:.2f}', 'Total Return (%)': '{:.2f}', 'Volatility (%)': '{:.2f}',
            'Sharpe Ratio': '{:.3f}', 'Max Drawdown (%)': '{:.2f}',
            'Start Price ($)': '{:.2f}', 'End Price ($)': '{:.2f}'
        }, na_rep="N/A").map(
            lambda val: 'color: green' if isinstance(val, (int, float)) and val > 0 else 'color: red' if isinstance(val, (int, float)) and val < 0 else '',
            subset=['Total Return (%)', 'Sharpe Ratio']
        ).map(
            lambda val: 'color: red' if isinstance(val, (int, float)) and val < 0 else '',
            subset=['Max Drawdown (%)']
        )
        
        st.dataframe(styled_df, use_container_width=True)


def create_volume_chart(filtered_stock_data):
    """Create a volume chart from pre-filtered data."""
    if not filtered_stock_data:
        return
    
    st.subheader("📊 Trading Volume")
    
    selected_ticker = st.selectbox(
        "Select ticker for volume chart:",
        list(filtered_stock_data.keys())
    )
    
    if selected_ticker and selected_ticker in filtered_stock_data:
        data = filtered_stock_data[selected_ticker]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data.index, y=data['Volume'], name='Volume',
            marker_color='lightblue',
            hovertemplate='<b>Volume</b><br>Date: %{x|%Y-%m-%d}<br>Volume: %{y:,.0f}<extra></extra>'
        ))
        
        timeframe_labels = {'1M': '1 Month', '3M': '3 Months', '6M': '6 Months', '1Y': '1 Year', '2Y': '2 Years', '3Y': '3 Years', '5Y': '5 Years'}
        timeframe_label = f" - {timeframe_labels[st.session_state.selected_timeframe]}"
        
        fig.update_layout(
            title=f"Trading Volume - {selected_ticker}{timeframe_label}",
            xaxis_title="Date", yaxis_title="Volume",
            template="plotly_white", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def perform_automatic_cleanup():
    """Perform automatic database cleanup if needed."""
    now = datetime.now()
    if (st.session_state.last_cleanup_check is None or now - st.session_state.last_cleanup_check > timedelta(hours=1)):
        st.session_state.last_cleanup_check = now
        st.session_state.db.cleanup_old_data(max_size_mb=50, max_age_days=180)


def main():
    """Main application function."""
    st.set_page_config(page_title="Stock Performance Tool", page_icon="📈", layout="wide")
    
    initialize_session_state()
    perform_automatic_cleanup()
    
    st.title("📈 Stock Performance Tool")
    st.markdown("Analyze your favorite stocks with interactive charts and performance metrics.")
    
    tickers, fetch_button = create_sidebar()
    timeframes, days_in_period = create_timeframe_buttons()

    if st.session_state.show_instructions:
        with st.container():
            col1, col2, col3 = st.columns([1, 8, 1])
            with col2:
                show_how_to_use_content()
                if st.button("✖️ Close Instructions", use_container_width=True):
                    st.session_state.show_instructions = False
                    st.rerun()
        return

    # --- Optimized Data Fetching Logic ---

    # 1. Determine if a primary fetch is needed (change in tickers, first load, manual button)
    primary_fetch_needed = False
    if fetch_button and tickers:
        primary_fetch_needed = True
    elif st.session_state.first_load and tickers:
        primary_fetch_needed = True
        st.session_state.first_load = False
    elif set(tickers) != set(st.session_state.current_tickers):
        primary_fetch_needed = True

    if primary_fetch_needed:
        st.session_state.current_tickers = tickers
        
        # --- Check for 1-Year Daily data in cache before fetching ---
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        
        is_daily_data_fresh = st.session_state.db.check_batch_data_freshness(
            tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), 'daily'
        )
        
        if is_daily_data_fresh:
            # Load from DB if fresh
            df = st.session_state.db.get_stock_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), 'daily')
            daily_data = {ticker: df[df['ticker'] == ticker].set_index('date') for ticker in tickers}
            st.session_state.daily_data = daily_data
        else:
            # Fetch from API if not fresh
            timeframe_config = {'days': 365, 'label': '1Y'}
            progress_callback, progress_bar, status_text = create_progress_callback()
            result, _, _, failed_tickers = st.session_state.data_fetcher.fetch_stocks_with_timeframe(
                tickers, timeframe_config, st.session_state.db, progress_callback
            )
            progress_bar.empty(); status_text.empty()
            st.session_state.daily_data = result
            if failed_tickers:
                st.warning(f"Failed to load daily data for: {', '.join(failed_tickers)}")
        
        st.session_state.weekly_data = {}  # Clear weekly data on new primary fetch

    # 2. Handle Long-Term View (2Y, 3Y, 5Y)
    is_long_term_view = days_in_period >= 730
    if is_long_term_view and tickers and not st.session_state.weekly_data:
        
        # --- Check for 5-Year Weekly data in cache before fetching ---
        end_date = date.today()
        start_date = end_date - timedelta(days=1825) # Always check for 5 years
        
        is_weekly_data_fresh = st.session_state.db.check_batch_data_freshness(
            tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), 'weekly'
        )

        if is_weekly_data_fresh:
            # Load from DB if fresh
            df = st.session_state.db.get_stock_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), 'weekly')
            weekly_data = {ticker: df[df['ticker'] == ticker].set_index('date') for ticker in tickers}
            st.session_state.weekly_data = weekly_data
        else:
            # Fetch 5 years of weekly data from API if not fresh
            timeframe_config = {'days': 1825, 'label': '5Y'}
            progress_callback, progress_bar, status_text = create_progress_callback()
            result, _, _, failed_tickers = st.session_state.data_fetcher.fetch_stocks_with_timeframe(
                tickers, timeframe_config, st.session_state.db, progress_callback
            )
            progress_bar.empty(); status_text.empty()
            st.session_state.weekly_data = result
            if failed_tickers:
                st.warning(f"Failed to load weekly data for: {', '.join(failed_tickers)}")

    # 3. Determine which data to display
    data_to_display = st.session_state.weekly_data if is_long_term_view else st.session_state.daily_data

    # Filter the data ONCE, and pass the filtered data to all components
    if data_to_display:
        filtered_data = filter_data_by_timeframe(data_to_display, days_in_period)
        
        if filtered_data:
            create_price_chart(filtered_data)
            create_performance_table(filtered_data)
            create_volume_chart(filtered_data)
        else:
            st.warning("No data available for the selected timeframe.")
    elif not tickers:
        st.info("Enter stock tickers in the sidebar to get started.")

if __name__ == "__main__":
    main() 