import os
import yfinance as yf
import pandas as pd

def fetch_data(tickers=["TSLA", "BND", "SPY"], start_date="2015-01-01", end_date="2026-06-30"):
    """
    Fetches historical financial data from yfinance for the given tickers and date range.
    """
    print(f"Fetching data for tickers: {tickers} from {start_date} to {end_date}...")
    data_dict = {}
    for ticker in tickers:
        asset_data = yf.download(ticker, start=start_date, end=end_date)
        if asset_data.empty:
            raise ValueError(f"No data returned for ticker {ticker}.")
        
        # Flatten MultiIndex columns if present (common in newer yfinance versions)
        if isinstance(asset_data.columns, pd.MultiIndex):
            asset_data.columns = asset_data.columns.get_level_values(0)
            
        data_dict[ticker] = asset_data
    return data_dict

def clean_data(data_dict):
    """
    Cleans the data by checking for missing values, formatting column types,
    and ensuring proper indexing.
    """
    cleaned_dict = {}
    for ticker, df in data_dict.items():
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        
        # If 'Adj Close' is not in columns, copy from 'Close'
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
            
        # Select key columns
        df_cleaned = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
        
        # Forward fill and backward fill any missing values
        df_cleaned.ffill(inplace=True)
        df_cleaned.bfill(inplace=True)
        
        cleaned_dict[ticker] = df_cleaned
    return cleaned_dict

def save_data(data_dict, output_dir="portfolio-optimization/data/processed"):
    """
    Saves cleaned asset dataframes to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    for ticker, df in data_dict.items():
        filepath = os.path.join(output_dir, f"{ticker}_cleaned.csv")
        df.to_csv(filepath)
        print(f"Saved {ticker} data to {filepath}")

def load_local_data(tickers=["TSLA", "BND", "SPY"], input_dir="portfolio-optimization/data/processed"):
    """
    Loads saved cleaned dataframes.
    """
    data_dict = {}
    for ticker in tickers:
        filepath = os.path.join(input_dir, f"{ticker}_cleaned.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cleaned file not found for {ticker} at {filepath}")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        data_dict[ticker] = df
    return data_dict
