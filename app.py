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
    
    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    
    # Preset date options
    date_option = st.sidebar.selectbox(
        "Select time period:",
        ["Custom", "1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"]
    )
    
    end_date = date.today()
    
    if date_option == "Custom":
        start_date = st.sidebar.date_input(
            "Start Date:",
            value=end_date - timedelta(days=365),
            max_value=end_date
        )
        end_date = st.sidebar.date_input(
            "End Date:",
            value=end_date,
            max_value=end_date
        )
    else:
        days_mapping = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730,
            "5 Years": 1825
        }
        days = days_mapping[date_option]
        start_date = end_date - timedelta(days=days)
    
    # Fetch data button
    fetch_button = st.sidebar.button(
        "📊 Fetch Stock Data", 
        type="primary",
        use_container_width=True
    )
    
    # Database management
    st.sidebar.subheader("🗄️ Database")
    available_tickers = st.session_state.db.get_available_tickers()
    
    if available_tickers:
        st.sidebar.markdown(f"**Stored Tickers:** {len(available_tickers)}")
        selected_to_delete = st.sidebar.selectbox(
            "Delete ticker data:",
            ["Select ticker..."] + available_tickers
        )
        
        if st.sidebar.button("🗑️ Delete Selected", type="secondary"):
            if selected_to_delete != "Select ticker...":
                st.session_state.db.delete_stock_data(selected_to_delete)
                st.sidebar.success(f"Deleted {selected_to_delete}")
                st.rerun()
    else:
        st.sidebar.info("No data stored yet")
    
    return tickers, start_date, end_date, fetch_button


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


def create_price_chart(stock_data_dict):
    """Create an interactive price chart."""
    if not stock_data_dict:
        return
    
    st.subheader("📊 Stock Performance Comparison")
    
    # Create percentage change data for comparison
    percentage_data = st.session_state.data_fetcher.normalize_prices(stock_data_dict)
    
    if percentage_data.empty:
        st.warning("No data available for charting")
        return
    
    # Create plotly figure
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, ticker in enumerate(percentage_data.columns):
        fig.add_trace(go.Scatter(
            x=percentage_data.index,
            y=percentage_data[ticker],
            mode='lines',
            name=ticker,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f'<b>{ticker}</b><br>Date: %{{x}}<br>Change: %{{y:.1f}}%<extra></extra>'
        ))
    
    # Add a horizontal line at 0% for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Stock Performance Comparison (% Change from Start)",
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
    
    # Select ticker for volume chart
    selected_ticker = st.selectbox(
        "Select ticker for volume chart:",
        list(stock_data_dict.keys())
    )
    
    if selected_ticker and selected_ticker in stock_data_dict:
        data = stock_data_dict[selected_ticker]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='lightblue',
            hovertemplate='<b>Volume</b><br>Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Trading Volume - {selected_ticker}",
            xaxis_title="Date",
            yaxis_title="Volume",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


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
    
    # Main title
    st.title("📈 Stock Portfolio Tracker")
    st.markdown("Track and analyze your favorite stocks with interactive charts and performance metrics.")
    
    # Create sidebar
    tickers, start_date, end_date, fetch_button = create_sidebar()
    
    # Main content area
    if fetch_button and tickers:
        if len(tickers) > 10:
            st.error("Please enter no more than 10 stock tickers.")
        else:
            # Store status messages for display at bottom
            st.session_state.fetch_status = f"Fetching data for: {', '.join(tickers)}"
            st.session_state.show_fetch_status = True
            
            # Create progress tracking
            progress_callback, progress_bar, status_text = create_progress_callback()
            
            # Fetch stock data
            result = st.session_state.data_fetcher.fetch_multiple_stocks(
                tickers, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d'),
                progress_callback
            )
            
            # Unpack the result
            stock_data, successful_tickers, failed_tickers = result
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Store fetch result messages for display at bottom
            if successful_tickers:
                st.session_state.fetch_success_status = f"Successfully fetched data for: {', '.join(successful_tickers)}"
                st.session_state.show_fetch_success_status = True
            
            if failed_tickers:
                st.session_state.fetch_failed_status = f"Failed to fetch data for: {', '.join(failed_tickers)}"
                st.session_state.show_fetch_failed_status = True
            
            if stock_data:
                # Store data in database
                with st.spinner("Saving data to database..."):
                    for ticker, data in stock_data.items():
                        try:
                            st.session_state.db.insert_stock_data(ticker, data)
                        except Exception as e:
                            st.error(f"Error saving {ticker} to database: {str(e)}")
                
                # Store in session state for immediate display
                st.session_state.stock_data = stock_data
                
                # Store success status for display at bottom
                st.session_state.success_status = "Data fetched and saved successfully!"
                st.session_state.show_success_status = True
    
    # Display data if available
    if st.session_state.stock_data:
        
        # Main price chart (full width for better visibility)
        create_price_chart(st.session_state.stock_data)
        
        # Performance metrics table with stock information (full width between charts)
        create_performance_table(st.session_state.stock_data)
        
        # Volume chart (full width at bottom)
        create_volume_chart(st.session_state.stock_data)
        
        # Display status messages at the bottom
        if hasattr(st.session_state, 'show_fetch_status') and st.session_state.show_fetch_status:
            st.info(st.session_state.fetch_status)
            st.session_state.show_fetch_status = False
        
        if hasattr(st.session_state, 'show_fetch_success_status') and st.session_state.show_fetch_success_status:
            st.success(st.session_state.fetch_success_status)
            st.session_state.show_fetch_success_status = False
        
        if hasattr(st.session_state, 'show_fetch_failed_status') and st.session_state.show_fetch_failed_status:
            st.warning(st.session_state.fetch_failed_status)
            st.session_state.show_fetch_failed_status = False
        
        if hasattr(st.session_state, 'show_success_status') and st.session_state.show_success_status:
            st.success(st.session_state.success_status)
            st.session_state.show_success_status = False
    
    else:
        # Welcome message
        st.markdown("""
        ### Welcome to Stock Portfolio Tracker! 🎯
        
        **How to get started:**
        1. Enter stock ticker symbols in the sidebar (e.g., AAPL, GOOGL, MSFT)
        2. Select your desired time period
        3. Click "Fetch Stock Data" to retrieve and store the data
        4. View interactive charts and performance metrics
        
        **Features:**
        - 📊 Interactive price comparison charts
        - 📈 Performance metrics and analysis
        - 🗄️ Local database storage for offline access
        - 📱 Responsive design for mobile and desktop
        
        **Tips:**
        - You can track up to 10 stocks at once
        - Data is stored locally in SQLite for fast access
        - Use the database section to manage stored data
        """)
        
        # Show some popular tickers as examples
        popular_tickers = st.session_state.data_fetcher.get_popular_tickers()
        st.markdown("**Popular Stock Tickers:**")
        
        # Create columns for ticker buttons
        cols = st.columns(5)
        for i, ticker in enumerate(popular_tickers[:10]):
            with cols[i % 5]:
                if st.button(f"📊 {ticker}", key=f"popular_{ticker}"):
                    # Add ticker to the input field
                    current_input = st.session_state.ticker_input.strip()
                    if current_input:
                        # Add to existing tickers if not already present
                        existing_tickers = [t.strip().upper() for t in current_input.split(',') if t.strip()]
                        if ticker not in existing_tickers:
                            st.session_state.ticker_input = current_input + f", {ticker}"
                        else:
                            st.info(f"{ticker} is already in your list!")
                    else:
                        # Set as first ticker
                        st.session_state.ticker_input = ticker
                    st.rerun()


if __name__ == "__main__":
    main() 