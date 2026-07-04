import numpy as np
import pandas as pd

def run_backtest(returns_df, weights_dict, initial_capital=10000.0, rebalance_monthly=False):
    """
    Backtests a portfolio strategy given daily returns and weights.
    Supports either buy-and-hold (no rebalancing) or monthly rebalancing.
    """
    assets = list(weights_dict.keys())
    weights = np.array([weights_dict[a] for a in assets])
    
    # Align returns
    daily_returns = returns_df[assets].copy()
    
    if not rebalance_monthly:
        # Buy and hold strategy
        # Calculate daily portfolio return
        portfolio_returns = daily_returns.dot(weights)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        portfolio_value = initial_capital * cumulative_returns
        # Prepend initial capital
        portfolio_value = pd.Series([initial_capital] + portfolio_value.tolist(), 
                                    index=[daily_returns.index[0] - pd.Timedelta(days=1)] + list(daily_returns.index))
    else:
        # Monthly rebalancing strategy
        portfolio_value = []
        current_val = initial_capital
        
        # Group by year and month
        groups = daily_returns.groupby([daily_returns.index.year, daily_returns.index.month])
        
        for (year, month), group_df in groups:
            # Rebalance at the beginning of each month: allocate current_val according to weights
            asset_vals = current_val * weights
            # Track daily value in this month
            for idx, date in enumerate(group_df.index):
                # Apply daily return to each asset
                asset_returns = group_df.loc[date].values
                asset_vals = asset_vals * (1 + asset_returns)
                current_val = np.sum(asset_vals)
                portfolio_value.append((date, current_val))
            # After the month ends, the final current_val is carried over
            
        portfolio_value_df = pd.DataFrame(portfolio_value, columns=['Date', 'Value']).set_index('Date')
        portfolio_value = pd.Series([initial_capital] + portfolio_value_df['Value'].tolist(), 
                                    index=[daily_returns.index[0] - pd.Timedelta(days=1)] + list(portfolio_value_df.index))
        
    portfolio_returns = portfolio_value.pct_change().dropna()
    return portfolio_value, portfolio_returns

def calculate_backtest_metrics(portfolio_value, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculates final performance metrics: Total Return, Annualized Return, Sharpe Ratio, Max Drawdown.
    """
    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
    
    # Calculate days
    n_days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
    years = n_days / 365.25
    annualized_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    
    # Daily returns for Sharpe and Drawdown
    daily_returns = portfolio_value.pct_change().dropna()
    
    # Volatility
    daily_vol = daily_returns.std()
    annualized_vol = daily_vol * np.sqrt(periods_per_year)
    
    # Sharpe Ratio
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / annualized_vol if annualized_vol > 0 else 0
    
    # Maximum Drawdown
    roll_max = portfolio_value.cummax()
    drawdown = (portfolio_value - roll_max) / roll_max
    max_drawdown = drawdown.min()
    
    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }
