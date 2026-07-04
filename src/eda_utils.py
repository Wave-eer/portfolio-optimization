import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def calculate_daily_returns(df, price_col='Adj Close'):
    """
    Calculates daily returns (percentage change) for an asset.
    """
    returns = df[price_col].pct_change().dropna()
    return returns

def calculate_rolling_stats(df, window=20, price_col='Adj Close'):
    """
    Calculates rolling mean and rolling standard deviation (volatility).
    """
    rolling_mean = df[price_col].rolling(window=window).mean()
    rolling_std = df[price_col].rolling(window=window).std()
    return rolling_mean, rolling_std

def detect_outliers(returns, threshold=3.0):
    """
    Detects outliers in daily returns using Z-score threshold.
    """
    mean = returns.mean()
    std = returns.std()
    z_scores = (returns - mean) / std
    outliers = returns[np.abs(z_scores) > threshold]
    return outliers

def run_adf_test(series):
    """
    Runs the Augmented Dickey-Fuller test for stationarity.
    """
    result = adfuller(series.dropna())
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    is_stationary = p_value < 0.05
    
    return {
        "adf_stat": adf_stat,
        "p_value": p_value,
        "critical_values": critical_values,
        "is_stationary": is_stationary
    }

def calculate_var(returns, confidence_level=0.95):
    """
    Calculates historical Value at Risk (VaR).
    """
    # VaR represents the minimum loss expected with a certain confidence
    cutoff = 1 - confidence_level
    var = -np.percentile(returns, cutoff * 100)
    return var

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculates the historical annualized Sharpe Ratio.
    """
    mean_return = returns.mean()
    std_return = returns.std()
    if std_return == 0:
        return 0.0
    
    excess_return = mean_return - (risk_free_rate / periods_per_year)
    daily_sharpe = excess_return / std_return
    annualized_sharpe = daily_sharpe * np.sqrt(periods_per_year)
    return annualized_sharpe
