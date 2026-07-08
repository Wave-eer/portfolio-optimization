import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Ensure the parent directory is in sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our custom modules
from src.data_loader import fetch_data, clean_data, save_data, load_local_data
from src.eda_utils import (
    calculate_daily_returns, calculate_rolling_stats, detect_outliers,
    run_adf_test, calculate_var, calculate_sharpe_ratio,
    calculate_cvar, calculate_max_drawdown, calculate_sortino_ratio,
    calculate_skewness, calculate_kurtosis, run_jarque_bera_test
)
from src.models import (
    calculate_metrics, fit_auto_arima, forecast_arima,
    prepare_lstm_data, train_lstm, forecast_lstm_test, forecast_lstm_future
)
from src.portfolio import (
    optimize_portfolio, generate_efficient_frontier, get_portfolio_metrics
)
from src.backtest import run_backtest, calculate_backtest_metrics

# Standard library and third party imports
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import norm


def main():
    # 1. Setup paths
    base_dir = "."
    data_dir = os.path.join(base_dir, "data", "processed")
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # 2. Extract and Clean Data
    print("--- STEP 1: Data Ingestion & Cleaning ---")
    tickers = ["TSLA", "BND", "SPY"]
    try:
        raw_data = fetch_data(tickers, start_date="2015-01-01", end_date="2026-06-30")
        cleaned_data = clean_data(raw_data)
        save_data(cleaned_data, output_dir=data_dir)
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Attempting to load local data if exists...")
        cleaned_data = load_local_data(tickers, input_dir=data_dir)
        
    # 3. Exploratory Data Analysis (EDA)
    print("\n--- STEP 2: Exploratory Data Analysis ---")
    eda_summary = {}
    
    # Generate Plots
    # Plot 1: Closing Prices
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        plt.plot(cleaned_data[ticker]['Adj Close'], label=ticker)
    plt.title("Adjusted Closing Prices (2015 - 2026)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "closing_prices.png"))
    plt.close()
    
    # Calculate Returns
    returns_dict = {}
    for ticker in tickers:
        returns_dict[ticker] = calculate_daily_returns(cleaned_data[ticker])
        
    # Plot 2: Daily Percentage Change (Returns)
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        plt.plot(returns_dict[ticker], label=ticker, alpha=0.6)
    plt.title("Daily Percentage Change (Returns)")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "daily_returns.png"))
    plt.close()
    
    # Rolling Volatility
    plt.figure(figsize=(12, 6))
    for ticker in tickers:
        _, rolling_std = calculate_rolling_stats(cleaned_data[ticker], window=20)
        plt.plot(rolling_std, label=f"{ticker} 20d Rolling Vol")
    plt.title("Rolling Volatility (20-day Standard Deviation)")
    plt.xlabel("Date")
    plt.ylabel("Standard Deviation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "rolling_volatility.png"))
    plt.close()

    # Normality Diagnostics: Return Distribution & Boxplots
    for ticker in tickers:
        rets = returns_dict[ticker]
        
        # Histograms + KDE + Normal fit overlay
        plt.figure(figsize=(10, 5))
        sns.histplot(rets, kde=True, stat="density", label="Historical Returns", bins=50, alpha=0.6)
        
        # Fit normal distribution
        mu, std = norm.fit(rets)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'r--', linewidth=2, label=f"Normal Fit (mu={mu:.4f}, std={std:.4f})")
        
        plt.title(f"{ticker} Returns Distribution vs. Normal Curve")
        plt.xlabel("Daily Return")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{ticker}_returns_distribution.png"))
        plt.close()
        
        # Boxplot
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=rets)
        plt.title(f"{ticker} Returns Boxplot (Outlier Analysis)")
        plt.ylabel("Daily Return")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{ticker}_returns_boxplot.png"))
        plt.close()

    # Autocorrelation (ACF / PACF) Analysis
    for ticker in tickers:
        rets = returns_dict[ticker]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        plot_acf(rets, lags=40, ax=ax1, title=f"{ticker} ACF (Returns)")
        plot_pacf(rets, lags=40, ax=ax2, title=f"{ticker} PACF (Returns)")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{ticker}_acf_pacf.png"))
        plt.close()

    # Time Series Decomposition (Trend, Seasonal, Residual)
    # Daily stock data has a lot of noise, decompose with period=252 (annual business days)
    for ticker in tickers:
        prices = cleaned_data[ticker]['Adj Close']
        # Resample to daily frequency and fill gaps to prevent statsmodels errors on missing business days
        prices_resampled = prices.asfreq('D').ffill()
        result = seasonal_decompose(prices_resampled, model='additive', period=365)
        
        fig = result.plot()
        fig.set_size_inches(12, 8)
        plt.suptitle(f"{ticker} Price Additive Decomposition (365d Period)", y=1.02, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{ticker}_decomposition.png"))
        plt.close()

    # Calculate Stationarity & Comprehensive Risk Metrics
    for ticker in tickers:
        adj_close = cleaned_data[ticker]['Adj Close']
        rets = returns_dict[ticker]
        
        adf_close = run_adf_test(adj_close)
        adf_rets = run_adf_test(rets)
        
        var_95 = calculate_var(rets, confidence_level=0.95)
        var_99 = calculate_var(rets, confidence_level=0.99)
        cvar_95 = calculate_cvar(rets, confidence_level=0.95)
        cvar_99 = calculate_cvar(rets, confidence_level=0.99)
        
        max_dd = calculate_max_drawdown(adj_close)
        sharpe = calculate_sharpe_ratio(rets)
        sortino = calculate_sortino_ratio(rets)
        
        skewness = calculate_skewness(rets)
        kurtosis = calculate_kurtosis(rets)
        jb_res = run_jarque_bera_test(rets)
        
        outliers = detect_outliers(rets, threshold=3.0)
        
        eda_summary[ticker] = {
            "ADF_Close_p_val": float(adf_close["p_value"]),
            "ADF_Close_Stationary": bool(adf_close["is_stationary"]),
            "ADF_Returns_p_val": float(adf_rets["p_value"]),
            "ADF_Returns_Stationary": bool(adf_rets["is_stationary"]),
            "Value_at_Risk_95": float(var_95),
            "Value_at_Risk_99": float(var_99),
            "Conditional_VaR_95": float(cvar_95),
            "Conditional_VaR_99": float(cvar_99),
            "Max_Drawdown": float(max_dd),
            "Sharpe_Ratio": float(sharpe),
            "Sortino_Ratio": float(sortino),
            "Skewness": float(skewness),
            "Kurtosis": float(kurtosis),
            "Jarque_Bera_stat": float(jb_res["jb_stat"]),
            "Jarque_Bera_p_val": float(jb_res["p_value"]),
            "Jarque_Bera_normal": bool(jb_res["is_normal"]),
            "Num_Outliers_3std": int(len(outliers))
        }
        
    print(json.dumps(eda_summary, indent=4))
    with open(os.path.join(data_dir, "eda_summary.json"), "w") as f:
        json.dump(eda_summary, f, indent=4)
        
    # 4. Time Series Modeling (Task 2 & 3)
    print("\n--- STEP 3: Time Series Forecasting for TSLA ---")
    tsla_data = cleaned_data["TSLA"]
    tsla_close = tsla_data['Adj Close']
    
    # Chronological Split (Train: 2015-2024, Test: 2025-2026)
    train_split_date = "2025-01-01"
    train_close = tsla_close[:train_split_date]
    test_close = tsla_close[train_split_date:]
    
    print(f"Train set: {train_close.index[0].date()} to {train_close.index[-1].date()} ({len(train_close)} days)")
    print(f"Test set: {test_close.index[0].date()} to {test_close.index[-1].date()} ({len(test_close)} days)")
    
    # ARIMA Model
    # Fit ARIMA (since it takes a bit of time, we can use a simpler model configuration if auto_arima is slow, or standard orders)
    try:
        arima_model = fit_auto_arima(train_close, m=1)
        arima_order = arima_model.order
        arima_seasonal_order = arima_model.seasonal_order
    except Exception as e:
        print(f"Auto ARIMA fitting failed: {e}. Using fallback ARIMA(1, 1, 1)")
        arima_order = (1, 1, 1)
        arima_seasonal_order = (0, 0, 0, 0)
        
    # Fit statsmodels SARIMAX with selected parameters for test period forecasting
    model_sm = SARIMAX(train_close, order=arima_order, seasonal_order=arima_seasonal_order)
    fitted_sm = model_sm.fit(disp=False)
    
    # Predict test period
    arima_test_preds = fitted_sm.forecast(steps=len(test_close))
    arima_test_preds.index = test_close.index
    
    # LSTM Model
    # Scale and prepare sequence data
    window_size = 60
    X_train, y_train, scaler = prepare_lstm_data(train_close, window_size)
    # Train LSTM
    lstm_model, device = train_lstm(X_train, y_train, epochs=20, batch_size=32, lr=0.001)
    
    # Test set preparation for LSTM
    # We need the last 60 days of the training set to start predicting the test set
    lstm_full_series = pd.concat([train_close[-window_size:], test_close])
    X_test, y_test, _ = prepare_lstm_data(lstm_full_series, window_size)
    
    # Predict test period
    lstm_test_preds = forecast_lstm_test(lstm_model, X_test, scaler, device)
    lstm_test_preds = pd.Series(lstm_test_preds, index=test_close.index)
    
    # Evaluate
    arima_metrics = calculate_metrics(test_close, arima_test_preds)
    lstm_metrics = calculate_metrics(test_close, lstm_test_preds)
    
    metrics_df = pd.DataFrame({
        "ARIMA": arima_metrics,
        "LSTM": lstm_metrics
    })
    print("\nModel Evaluation Metrics on Test Set:")
    print(metrics_df)
    metrics_df.to_csv(os.path.join(data_dir, "model_comparison.csv"))
    
    # Plot 3: Model Predictions vs Actual
    plt.figure(figsize=(12, 6))
    plt.plot(train_close[-200:], label="Train Actual (last 200 days)")
    plt.plot(test_close, label="Test Actual")
    plt.plot(arima_test_preds, label="ARIMA Prediction")
    plt.plot(lstm_test_preds, label="LSTM Prediction")
    plt.title("TSLA Stock Price Forecasting Comparison")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "model_comparison.png"))
    plt.close()
    
    # 5. Future Forecasting (6 Months / 126 Trading Days)
    print("\n--- STEP 4: Future Forecasting ---")
    forecast_steps = 126 # ~6 months
    
    # Refit ARIMA on full dataset for future forecast
    model_sm_full = SARIMAX(tsla_close, order=arima_order, seasonal_order=arima_seasonal_order)
    fitted_sm_full = model_sm_full.fit(disp=False)
    
    # Forecast ARIMA future
    arima_future_forecast = fitted_sm_full.get_forecast(steps=forecast_steps)
    arima_future_mean = arima_future_forecast.predicted_mean
    arima_future_ci = arima_future_forecast.conf_int(alpha=0.05) # 95% CI
    
    # Forecast LSTM future
    # We need the last 60 days of the full dataset
    last_sequence_scaled = scaler.transform(tsla_close[-window_size:].values.reshape(-1, 1))
    lstm_future_mean = forecast_lstm_future(lstm_model, last_sequence_scaled, forecast_steps, scaler, device)
    
    # Generate future dates index (business days)
    future_dates = pd.date_range(start=tsla_close.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')
    
    arima_future_mean.index = future_dates
    arima_future_ci.index = future_dates
    lstm_future_mean = pd.Series(lstm_future_mean, index=future_dates)
    
    # Save Future Forecasts
    future_df = pd.DataFrame({
        "ARIMA_Mean": arima_future_mean,
        "ARIMA_Lower_CI": arima_future_ci.iloc[:, 0],
        "ARIMA_Upper_CI": arima_future_ci.iloc[:, 1],
        "LSTM_Mean": lstm_future_mean
    })
    future_df.to_csv(os.path.join(data_dir, "future_forecast.csv"))
    
    # Plot 4: Future Forecast with Confidence Intervals
    plt.figure(figsize=(12, 6))
    plt.plot(tsla_close[-250:], label="Historical Actual (last Year)")
    plt.plot(arima_future_mean, label="ARIMA Future Forecast", color='blue')
    plt.fill_between(future_dates, arima_future_ci.iloc[:, 0], arima_future_ci.iloc[:, 1], color='blue', alpha=0.15, label="ARIMA 95% Confidence Interval")
    plt.plot(lstm_future_mean, label="LSTM Future Forecast", color='orange')
    plt.title("TSLA Future Stock Price Forecast (Next 6 Months)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "future_forecast.png"))
    plt.close()
    
    # Determine best model based on MAE/RMSE
    best_model_name = "ARIMA" if arima_metrics["RMSE"] < lstm_metrics["RMSE"] else "LSTM"
    print(f"Best performing model on test set: {best_model_name}")
    
    # 6. Portfolio Optimization (Task 4)
    print("\n--- STEP 5: Portfolio Optimization ---")
    
    # Expected returns
    # TSLA (Forecasted): use forecasted return. We can calculate the expected return as the average daily return of the forecast.
    # We will use the forecast mean from the best model.
    best_forecast_mean = arima_future_mean if best_model_name == "ARIMA" else lstm_future_mean
    forecast_returns = best_forecast_mean.pct_change().dropna()
    tsla_expected_daily_return = forecast_returns.mean()
    tsla_expected_annual_return = tsla_expected_daily_return * 252
    
    # BND and SPY (Historical): average daily returns annualized
    bnd_returns = returns_dict["BND"]
    spy_returns = returns_dict["SPY"]
    
    expected_returns_dict = {
        "TSLA": tsla_expected_annual_return,
        "BND": bnd_returns.mean() * 252,
        "SPY": spy_returns.mean() * 252
    }
    expected_returns_series = pd.Series(expected_returns_dict)
    
    # Covariance Matrix (annualized)
    # Combine returns into a single DataFrame
    combined_returns = pd.DataFrame({
        "TSLA": returns_dict["TSLA"],
        "BND": returns_dict["BND"],
        "SPY": returns_dict["SPY"]
    }).dropna()
    
    cov_matrix_annual = combined_returns.cov() * 252
    
    # Plot 5: Covariance Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(combined_returns.corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Asset Returns Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "correlation_heatmap.png"))
    plt.close()
    
    # Optimization
    weights_max_sharpe, weights_min_vol = optimize_portfolio(expected_returns_series, cov_matrix_annual)
    
    # Print Portfolios
    print("\nMax Sharpe Portfolio Weights:")
    print(weights_max_sharpe)
    print("Min Volatility Portfolio Weights:")
    print(weights_min_vol)
    
    # Efficient Frontier Plot
    results, weights_record = generate_efficient_frontier(expected_returns_series, cov_matrix_annual, num_portfolios=5000)
    
    # Extract metrics for optimized portfolios
    w_ms = [weights_max_sharpe[t] for t in tickers]
    ret_ms, vol_ms, sharpe_ms = get_portfolio_metrics(w_ms, expected_returns_series, cov_matrix_annual)
    
    w_mv = [weights_min_vol[t] for t in tickers]
    ret_mv, vol_mv, sharpe_mv = get_portfolio_metrics(w_mv, expected_returns_series, cov_matrix_annual)
    
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o', s=10, alpha=0.3)
    plt.colorbar(sc, label='Sharpe Ratio')
    plt.scatter(vol_ms, ret_ms, color='red', marker='*', s=200, label='Max Sharpe Ratio')
    plt.scatter(vol_mv, ret_mv, color='blue', marker='X', s=200, label='Min Volatility')
    plt.title("Efficient Frontier (TSLA, BND, SPY)")
    plt.xlabel("Annualized Volatility (Risk)")
    plt.ylabel("Annualized Expected Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "efficient_frontier.png"))
    plt.close()
    
    # Save portfolio summary
    portfolio_summary = {
        "Max_Sharpe": {
            "weights": weights_max_sharpe,
            "return": float(ret_ms),
            "volatility": float(vol_ms),
            "sharpe_ratio": float(sharpe_ms)
        },
        "Min_Volatility": {
            "weights": weights_min_vol,
            "return": float(ret_mv),
            "volatility": float(vol_mv),
            "sharpe_ratio": float(sharpe_mv)
        }
    }
    with open(os.path.join(data_dir, "portfolio_optimization.json"), "w") as f:
        json.dump(portfolio_summary, f, indent=4)
        
    # 7. Backtesting (Task 5)
    print("\n--- STEP 6: Strategy Backtesting ---")
    # Backtesting Period: 2025-01-01 to 2025-12-31 (or similar last year of training/test data)
    # We will use January 2025 - January 2026.
    backtest_returns = combined_returns.loc["2025-01-01":"2026-01-01"]
    
    # Strategy portfolio (Max Sharpe Ratio portfolio)
    strat_value, strat_returns = run_backtest(backtest_returns, weights_max_sharpe, rebalance_monthly=True)
    strat_metrics = calculate_backtest_metrics(strat_value)
    
    # Benchmark portfolio: 60% SPY / 40% BND
    benchmark_weights = {"TSLA": 0.0, "SPY": 0.6, "BND": 0.4}
    bench_value, bench_returns = run_backtest(backtest_returns, benchmark_weights, rebalance_monthly=False)
    bench_metrics = calculate_backtest_metrics(bench_value)
    
    print("\nBacktest Performance Comparison:")
    print("Strategy Portfolio Metrics:")
    print(json.dumps(strat_metrics, indent=4))
    print("Benchmark Portfolio Metrics (60/40 SPY/BND):")
    print(json.dumps(bench_metrics, indent=4))
    
    # Plot 6: Cumulative Returns
    plt.figure(figsize=(12, 6))
    plt.plot((strat_value / strat_value.iloc[0]) - 1, label="Optimized Strategy (Max Sharpe, Monthly Rebalanced)", color='blue')
    plt.plot((bench_value / bench_value.iloc[0]) - 1, label="Benchmark (60/40 SPY/BND, Buy & Hold)", color='grey', linestyle='--')
    plt.title("Cumulative Returns Comparison (Backtesting Period: 2025)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "backtest_cumulative_returns.png"))
    plt.close()
    
    # Save backtest results
    backtest_summary = {
        "Strategy_Metrics": strat_metrics,
        "Benchmark_Metrics": bench_metrics
    }
    with open(os.path.join(data_dir, "backtest_summary.json"), "w") as f:
        json.dump(backtest_summary, f, indent=4)
        
    print("\nPipeline run complete. All outputs saved in data/processed/ and plots/ directories.")

if __name__ == "__main__":
    main()
