import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns

def get_portfolio_metrics(weights, expected_returns_series, cov_matrix, risk_free_rate=0.0):
    """
    Calculates expected return, volatility, and Sharpe Ratio of a portfolio.
    """
    weights = np.array(weights)
    portfolio_return = np.dot(weights, expected_returns_series)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
    return portfolio_return, portfolio_volatility, sharpe_ratio

def optimize_portfolio(expected_returns_series, cov_matrix, risk_free_rate=0.0):
    """
    Calculates the Max Sharpe Ratio and Min Volatility portfolios using PyPortfolioOpt.
    """
    # Max Sharpe Portfolio
    ef_max_sharpe = EfficientFrontier(expected_returns_series, cov_matrix)
    try:
        raw_weights_max_sharpe = ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
        weights_max_sharpe = ef_max_sharpe.clean_weights()
    except Exception as e:
        print(f"Max Sharpe Optimization failed: {e}. Falling back to scipy optimization.")
        weights_max_sharpe = fallback_optimize(expected_returns_series, cov_matrix, objective='sharpe', rf=risk_free_rate)
        
    # Min Volatility Portfolio
    ef_min_vol = EfficientFrontier(expected_returns_series, cov_matrix)
    try:
        raw_weights_min_vol = ef_min_vol.min_volatility()
        weights_min_vol = ef_min_vol.clean_weights()
    except Exception as e:
        print(f"Min Volatility Optimization failed: {e}. Falling back to scipy optimization.")
        weights_min_vol = fallback_optimize(expected_returns_series, cov_matrix, objective='vol')

    return weights_max_sharpe, weights_min_vol

def generate_efficient_frontier(expected_returns_series, cov_matrix, num_portfolios=2000, risk_free_rate=0.0):
    """
    Generates random portfolios to construct the Efficient Frontier.
    """
    num_assets = len(expected_returns_series)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        p_return, p_vol, p_sharpe = get_portfolio_metrics(weights, expected_returns_series, cov_matrix, risk_free_rate)
        results[0, i] = p_return
        results[1, i] = p_vol
        results[2, i] = p_sharpe
        
    return results, weights_record

def fallback_optimize(expected_returns_series, cov_matrix, objective='sharpe', rf=0.0):
    """
    A scipy.optimize fallback if PyPortfolioOpt fails.
    """
    from scipy.optimize import minimize
    num_assets = len(expected_returns_series)
    
    def loss(weights):
        p_ret = np.dot(weights, expected_returns_series)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if objective == 'sharpe':
            return -(p_ret - rf) / p_vol if p_vol > 0 else 0
        else:
            return p_vol
            
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    init_weights = np.array([1.0 / num_assets] * num_assets)
    
    res = minimize(loss, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    asset_names = expected_returns_series.index
    return dict(zip(asset_names, res.x))
