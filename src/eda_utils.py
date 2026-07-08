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

def calculate_cvar(returns, confidence_level=0.95):
    """
    Calculates historical Conditional Value at Risk (CVaR) or Expected Shortfall.
    """
    cutoff = 1 - confidence_level
    var = calculate_var(returns, confidence_level=confidence_level)
    # Average of returns that are worse (lower) than the negative VaR
    # Since VaR is reported as positive number, lower is < -VaR
    worse_returns = returns[returns <= -var]
    if len(worse_returns) == 0:
        return var
    cvar = -worse_returns.mean()
    return cvar

def calculate_max_drawdown(prices):
    """
    Calculates maximum drawdown from a price series.
    Returns value as a percentage change (negative float).
    """
    if len(prices) == 0:
        return 0.0
    roll_max = prices.cummax()
    drawdown = (prices - roll_max) / roll_max
    return float(drawdown.min())

def calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculates the historical annualized Sortino Ratio (downside deviation risk-adjusted return).
    """
    mean_return = returns.mean()
    # Downside deviation uses only negative returns
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return 0.0
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    if downside_std == 0:
        return 0.0
    
    excess_return = mean_return - (risk_free_rate / periods_per_year)
    daily_sortino = excess_return / downside_std
    annualized_sortino = daily_sortino * np.sqrt(periods_per_year)
    return annualized_sortino

def calculate_skewness(returns):
    """
    Calculates the skewness of returns.
    """
    from scipy.stats import skew
    return float(skew(returns.dropna()))

def calculate_kurtosis(returns):
    """
    Calculates the excess kurtosis of returns.
    """
    from scipy.stats import kurtosis
    return float(kurtosis(returns.dropna()))

def run_jarque_bera_test(returns):
    """
    Runs the Jarque-Bera test for normality.
    Returns a dict with jb_stat, p_value, and whether returns are normal (p_value > 0.05).
    """
    from scipy.stats import jarque_bera
    stat, p_value = jarque_bera(returns.dropna())
    return {
        "jb_stat": float(stat),
        "p_value": float(p_value),
        "is_normal": bool(p_value > 0.05)
    }

