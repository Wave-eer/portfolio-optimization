# src/backtest.py
import numpy as np
import pandas as pd

def isolate_final_year(returns_df, years=1):
    """
    Split returns_df into (train_returns, test_returns) where test_returns is the final `years` years.
    Input: returns_df: DataFrame indexed by Datetime, columns = tickers, values = simple returns (not prices).
    """
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        raise ValueError("returns_df must have a DatetimeIndex")

    end = returns_df.index.max()
    start_test = end - pd.DateOffset(years=years) + pd.Timedelta(days=1)
    mask_test = returns_df.index >= start_test
    train = returns_df.loc[~mask_test].copy()
    test = returns_df.loc[mask_test].copy()
    return train, test

def _align_weights_to_index(weights, index, columns):
    """
    Align weights (dict/Series/DataFrame) to an index (returns index).
    - dict or Series -> expands to DataFrame repeating same row for each date in index
    - DataFrame -> reindexes to index with forward-fill (weights represent when they take effect)
    Returns DataFrame indexed by `index` and columns exactly matching `columns`.
    """
    if isinstance(weights, dict):
        wser = pd.Series(weights)
        wdf = pd.DataFrame([wser.reindex(columns).fillna(0).values], index=[index[0]], columns=columns)
        return pd.DataFrame(np.repeat(wdf.values, len(index), axis=0), index=index, columns=columns)

    if isinstance(weights, pd.Series):
        wdf = weights.reindex(columns).fillna(0)
        row = pd.DataFrame([wdf.values], index=[index[0]], columns=columns)
        return pd.DataFrame(np.repeat(row.values, len(index), axis=0), index=index, columns=columns)

    if isinstance(weights, pd.DataFrame):
        w = weights.reindex(columns, axis=1).copy()
        # reindex to the returns index using forward-fill so weight timestamps are effective dates
        w_reindexed = w.reindex(index).ffill().fillna(method='bfill').fillna(0)
        return w_reindexed

    raise ValueError("weights must be dict, pd.Series, or pd.DataFrame")

def run_backtest(returns_df, weights, initial_capital=10000.0, rebalance_monthly=False):
    """
    Backtests a portfolio given daily returns (DataFrame) and weights.
    - returns_df: DataFrame of daily returns (index=DatetimeIndex, columns = asset tickers)
    - weights: dict/Series/DataFrame describing target weights.
        * dict/Series -> static weights applied when rebalancing
        * DataFrame -> index are effective dates for weights (e.g., month-starts) and will be forward-filled to daily
    - rebalance_monthly: if False => buy-and-hold (single initial allocation); if True => rebalance at first trading day of each month
      Note: if weights is time-varying DataFrame, rebalancing happens to the weight row that is effective on that month's first date.
    Returns:
      - portfolio_value: pd.Series of portfolio value (includes a value at t0 equal to initial_capital)
      - portfolio_returns: pd.Series of simple returns (pct change) aligned with portfolio_value.index (first value is NaN, dropped)
    """
    # Ensure date index and assets
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        raise ValueError("returns_df must have a DatetimeIndex")

    assets = list(returns_df.columns)
    r = returns_df[assets].copy()

    # Align weights to daily index
    w_aligned = _align_weights_to_index(weights, r.index, assets)

    if not rebalance_monthly:
        # Buy and hold or static application of daily weights (weights change only when w_aligned changes)
        # Using target weights as fractions of portfolio at rebalancing times: daily portfolio return = sum(w_t * r_t)
        daily_port_rets = (w_aligned * r).sum(axis=1)
        cum = (1 + daily_port_rets).cumprod()
        portfolio_value = initial_capital * cum
        # prepend initial capital as day before first date to match previous behavior
        first_prior = r.index[0] - pd.Timedelta(days=1)
        portfolio_value = pd.Series(
            [initial_capital] + portfolio_value.tolist(),
            index=[first_prior] + list(portfolio_value.index)
        )
    else:
        # Monthly rebalancing: for each month, set dollar allocations to target weights effective at month start,
        # then let them grow with daily returns until next rebalance.
        values = []
        current_value = initial_capital

        # group rows by year-month in chronological order
        groups = r.groupby([r.index.year, r.index.month])
        for (year, month), group_df in groups:
            # choose the weight row that applies at the first trading day of the month
            first_date = group_df.index[0]
            target_weights = w_aligned.loc[first_date].values  # already has been ffilled to daily index
            # allocate dollars to each asset according to target weights
            asset_vals = current_value * target_weights
            # apply daily returns within the month
            for date in group_df.index:
                # update asset dollar values
                asset_returns = group_df.loc[date].values
                asset_vals = asset_vals * (1 + asset_returns)
                current_value = asset_vals.sum()
                values.append((date, current_value))

        portfolio_value_df = pd.DataFrame(values, columns=['date', 'value']).set_index('date')['value']
        # prepend initial capital
        first_prior = r.index[0] - pd.Timedelta(days=1)
        portfolio_value = pd.Series(
            [initial_capital] + portfolio_value_df.tolist(),
            index=[first_prior] + list(portfolio_value_df.index)
        )

    portfolio_returns = portfolio_value.pct_change().dropna()
    return portfolio_value, portfolio_returns
def evaluate_strategy_vs_benchmark(
    returns_df,
    strategy_weights,
    benchmark_weights,
    years=1,
    initial_capital=10000.0,
    rebalance_monthly=False,
    risk_free_rate=0.0,
    periods_per_year=252,
    plot=True,
):
    """
    Run strategy and benchmark on the final `years` of returns_df, compare cumulative returns and metrics.

    Parameters:
      - returns_df: DataFrame of daily simple returns (DatetimeIndex)
      - strategy_weights: dict/Series/DataFrame (see run_backtest)
      - benchmark_weights: dict/Series/DataFrame (e.g., {'SPY':0.6, 'BND':0.4})
      - years: number of final years to hold out / backtest on
      - initial_capital, rebalance_monthly, risk_free_rate, periods_per_year: forwarded to run_backtest/calc
      - plot: if True, show cumulative returns plot

    Returns:
      dict with keys:
        - 'test_returns' : DataFrame of test returns used
        - 'strategy': {'value': Series, 'returns': Series, 'metrics': dict}
        - 'benchmark': {'value': Series, 'returns': Series, 'metrics': dict}
    """
    import matplotlib.pyplot as plt

    # isolate final-year test set
    _, test_returns = isolate_final_year(returns_df, years=years)

    if test_returns.shape[0] == 0:
        raise ValueError("Test period (final year) contains no rows.")

    # run backtests (use same rebalance_monthly setting for both so comparisons are apples-to-apples)
    strat_value, strat_rets = run_backtest(test_returns, strategy_weights,
                                           initial_capital=initial_capital,
                                           rebalance_monthly=rebalance_monthly)
    bench_value, bench_rets = run_backtest(test_returns, benchmark_weights,
                                           initial_capital=initial_capital,
                                           rebalance_monthly=rebalance_monthly)

    # compute metrics
    strat_metrics = calculate_backtest_metrics(strat_value, risk_free_rate=risk_free_rate,
                                               periods_per_year=periods_per_year)
    bench_metrics = calculate_backtest_metrics(bench_value, risk_free_rate=risk_free_rate,
                                               periods_per_year=periods_per_year)

    # plot cumulative returns (normalize to 1 or initial_capital)
    if plot:
        plt.figure(figsize=(10, 6))
        (strat_value / strat_value.iloc[0]).plot(label='Strategy')
        (bench_value / bench_value.iloc[0]).plot(label='Benchmark')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Cumulative growth (normalized)')
        plt.title(f'Strategy vs Benchmark — final {years} year(s)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "test_returns": test_returns,
        "strategy": {"value": strat_value, "returns": strat_rets, "metrics": strat_metrics},
        "benchmark": {"value": bench_value, "returns": bench_rets, "metrics": bench_metrics},
    }


# Example usage (not run automatically):
# results = evaluate_strategy_vs_benchmark(
#     returns_df=all_returns,                       # your full returns DataFrame
#     strategy_weights=optimal_weights_from_task4,  # dict/Series/DataFrame from Task 4
#     benchmark_weights={'SPY': 0.6, 'BND': 0.4},
#     years=1,
#     initial_capital=10000.0,
#     rebalance_monthly=True,                       # or False for buy-and-hold
#     plot=True
# )
# print("Strategy metrics:", results['strategy']['metrics'])
# print("Benchmark metrics:", results['benchmark']['metrics'])
def calculate_backtest_metrics(portfolio_value, risk_free_rate=0.0, periods_per_year=252):
    """
    Compute:
      - Total Return
      - Annualized Return
      - Annualized Volatility
      - Sharpe Ratio (using risk_free_rate as annual)
      - Max Drawdown
    Expects portfolio_value: pd.Series indexed by date.
    """
    total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1.0

    # Years between first and last observation (use business-day count approx)
    n_days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
    years = n_days / 365.25 if n_days > 0 else 0.0
    annualized_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    daily_rets = portfolio_value.pct_change().dropna()
    daily_vol = daily_rets.std()
    annualized_vol = daily_vol * np.sqrt(periods_per_year)

    excess_return = annualized_return - risk_free_rate
    sharpe = excess_return / annualized_vol if annualized_vol > 0 else np.nan

    # max drawdown
    cumulative = portfolio_value.cummax()
    drawdown = (portfolio_value - cumulative) / cumulative
    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }
