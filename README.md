# Stock Performance Tool

An advanced and efficient Streamlit web application for real-time analysis and visualization of stocks, ETFs, and cryptocurrencies.

## Key Features

-   **Multi-Asset Analysis**: Track and compare up to 10 assets simultaneously, including stocks, ETFs, and cryptocurrencies.
-   **Interactive Performance Chart**: Visualize relative performance with a percentage-based chart where all assets start at 0% for easy comparison.
-   **Comprehensive Metrics**: View a detailed table of key performance indicators, including total return, volatility, Sharpe ratio, and max drawdown.
-   **Optimized Caching Strategy**: The application uses a smart caching layer that significantly minimizes API calls. Data is fetched from the local SQLite database whenever possible, making subsequent loads and analyses instantaneous.
-   **Hybrid Data Storage**: Employs an intelligent storage system that saves **daily** data for short-term views (≤ 1 year) and **weekly** data for long-term views (≥ 2 years), ensuring both high resolution and efficient storage.
-   **Intuitive UI**: A clean user interface with quick-add buttons for popular stocks, ETFs, and crypto makes ticker selection fast and easy.
-   **In-App Guidance**: A "How to Use" section provides clear instructions for all application features.

## Tech Stack

-   **Frontend**: Streamlit
-   **Data Source**: `yfinance` (Yahoo Finance API)
-   **Database/Caching**: SQLite
-   **Charting**: Plotly
-   **Data Manipulation**: pandas

## Project Structure

```
Proj_2_Stock_1/
├── app.py                 # Main Streamlit application & UI logic
├── stock_data.py          # Data fetching, processing, and normalization
├── database.py            # SQLite database operations and caching logic
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── data/
    └── stocks.db          # SQLite database file (created automatically)
```

## Local Setup

### 1. Create and Activate Virtual Environment

It's recommended to use a virtual environment to keep dependencies isolated:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

Ensure you have Python 3.8+ installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

### 4. Access the Application

Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

### 5. Deactivate Virtual Environment (when done)

```bash
deactivate
```

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