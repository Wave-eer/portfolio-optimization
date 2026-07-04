import numpy as np
import pandas as pd
import pytest
from src.eda_utils import (
    calculate_daily_returns, calculate_rolling_stats,
    run_adf_test, calculate_var, calculate_sharpe_ratio
)
from src.models import calculate_metrics
from src.portfolio import get_portfolio_metrics, optimize_portfolio

def test_eda_metrics():
    # Setup dummy series
    dates = pd.date_range(start="2026-01-01", periods=10)
    prices = pd.Series([100, 102, 101, 105, 104, 108, 107, 110, 112, 115], index=dates, name="Adj Close")
    df = pd.DataFrame(prices)
    
    returns = calculate_daily_returns(df)
    assert len(returns) == 9
    assert abs(returns.iloc[0] - 0.02) < 1e-4
    
    roll_mean, roll_std = calculate_rolling_stats(df, window=5)
    assert len(roll_mean) == 10
    assert pd.isna(roll_mean.iloc[3])
    assert not pd.isna(roll_mean.iloc[4])
    
    # Sharpe ratio
    sharpe = calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)
    
    # Value at Risk
    var = calculate_var(returns, confidence_level=0.95)
    assert isinstance(var, float)

def test_forecasting_metrics():
    y_true = [100, 102, 105, 103]
    y_pred = [98, 103, 104, 105]
    metrics = calculate_metrics(y_true, y_pred)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE" in metrics
    assert metrics["MAE"] > 0
    assert metrics["RMSE"] > 0
    assert metrics["MAPE"] > 0

def test_portfolio_optimization():
    expected_returns = pd.Series([0.15, 0.05, 0.08], index=["TSLA", "BND", "SPY"])
    cov_matrix = pd.DataFrame([
        [0.09, 0.001, 0.015],
        [0.001, 0.002, 0.0005],
        [0.015, 0.0005, 0.025]
    ], index=["TSLA", "BND", "SPY"], columns=["TSLA", "BND", "SPY"])
    
    w_ms, w_mv = optimize_portfolio(expected_returns, cov_matrix)
    
    # Verify weights sum to 1
    assert abs(sum(w_ms.values()) - 1.0) < 1e-4
    assert abs(sum(w_mv.values()) - 1.0) < 1e-4
    
    # Weights should be non-negative
    for k, v in w_ms.items():
        assert v >= -1e-4
    for k, v in w_mv.items():
        assert v >= -1e-4
