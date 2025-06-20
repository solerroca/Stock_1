# Stock Portfolio Tracker

A Streamlit web application for tracking and visualizing stock performance.

## Features
- Enter up to 10 stock tickers
- Fetch historical stock data from Yahoo Finance
- Store data in SQLite database
- Interactive charts showing stock performance
- Compare multiple stocks over time

## Project Structure
```
Proj_2_Stock_1/
├── app.py                 # Main Streamlit application
├── database.py           # Database operations
├── stock_data.py         # Stock data fetching and processing
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── data/                # Directory for SQLite database
    └── stocks.db        # SQLite database file (created automatically)
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Access the Application
Open your browser and go to `http://localhost:8501`

## How to Use
1. Enter stock tickers (e.g., AAPL, GOOGL, MSFT)
2. Select date range for historical data
3. Click "Fetch Data" to retrieve and store stock information
4. View interactive charts comparing stock performance

## Deployment
This app can be deployed on Streamlit Cloud for free. See deployment section for details.

## Technologies Used
- **Streamlit**: Web framework for Python
- **yfinance**: Yahoo Finance API for stock data
- **SQLite**: Lightweight database for data storage
- **Plotly**: Interactive charting library
- **Pandas**: Data manipulation and analysis 