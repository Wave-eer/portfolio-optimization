import json
import os

def create_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"Created notebook: {filename}")

def main():
    notebooks_dir = "notebooks"
    os.makedirs(notebooks_dir, exist_ok=True)
    
    # ---------------------------------------------
    # Notebook 1: eda_and_preprocess.ipynb
    # ---------------------------------------------
    cells_eda = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Task 1: Preprocess and Explore the Data\n",
                "\n",
                "This notebook extracts historical financial data for **TSLA**, **BND**, and **SPY** from yfinance covering the period from January 1, 2015 to June 30, 2026. We perform data cleaning, exploratory data analysis, stationarity testing, and calculate risk metrics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "sys.path.append('../')\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from src.data_loader import fetch_data, clean_data, save_data\n",
                "from src.eda_utils import (\n",
                "    calculate_daily_returns, calculate_rolling_stats, detect_outliers,\n",
                "    run_adf_test, calculate_var, calculate_sharpe_ratio,\n",
                "    calculate_cvar, calculate_max_drawdown, calculate_sortino_ratio,\n",
                "    calculate_skewness, calculate_kurtosis, run_jarque_bera_test\n",
                ")\n",
                "from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\n",
                "from statsmodels.tsa.seasonal import seasonal_decompose\n",
                "from scipy.stats import norm"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Extract Historical Financial Data\n",
                "We download the data using `yfinance` from Jan 1, 2015 to June 30, 2026."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tickers = [\"TSLA\", \"BND\", \"SPY\"]\n",
                "raw_data = fetch_data(tickers, start_date=\"2015-01-01\", end_date=\"2026-06-30\")\n",
                "cleaned_data = clean_data(raw_data)\n",
                "save_data(cleaned_data, output_dir=\"../data/processed\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Basic Data Statistics and Distributions"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for ticker in tickers:\n",
                "    print(f\"\\n--- Basic Statistics for {ticker} ---\")\n",
                "    print(cleaned_data[ticker].describe())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Exploratory Data Analysis (EDA)\n",
                "\n",
                "### A. Visualizing Closing Price over Time"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(14, 7))\n",
                "for ticker in tickers:\n",
                "    plt.plot(cleaned_data[ticker]['Adj Close'], label=ticker)\n",
                "plt.title(\"Adjusted Closing Price Over Time (2015 - 2026)\", fontsize=14)\n",
                "plt.xlabel(\"Date\", fontsize=12)\n",
                "plt.ylabel(\"Price ($)\", fontsize=12)\n",
                "plt.legend()\n",
                "plt.grid(True)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### B. Daily Percentage Change (Returns)\n",
                "Observing daily return volatility."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "returns_dict = {}\n",
                "plt.figure(figsize=(14, 7))\n",
                "for ticker in tickers:\n",
                "    returns_dict[ticker] = calculate_daily_returns(cleaned_data[ticker])\n",
                "    plt.plot(returns_dict[ticker], label=f\"{ticker} daily return\", alpha=0.6)\n",
                "plt.title(\"Daily Percentage Change (Returns) over Time\", fontsize=14)\n",
                "plt.xlabel(\"Date\", fontsize=12)\n",
                "plt.ylabel(\"Daily Return\", fontsize=12)\n",
                "plt.legend()\n",
                "plt.grid(True)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### C. Volatility Analysis (20-day Rolling Mean & Standard Deviation)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)\n",
                "for i, ticker in enumerate(tickers):\n",
                "    roll_mean, roll_std = calculate_rolling_stats(cleaned_data[ticker], window=20)\n",
                "    axes[i].plot(cleaned_data[ticker]['Adj Close'], label='Adj Close', color='blue')\n",
                "    axes[i].plot(roll_mean, label='20d Rolling Mean', color='orange')\n",
                "    axes[i].set_title(f\"{ticker} Price & Rolling Mean\", fontsize=12)\n",
                "    axes[i].legend()\n",
                "    axes[i].grid(True)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "plt.figure(figsize=(14, 5))\n",
                "for ticker in tickers:\n",
                "    _, roll_std = calculate_rolling_stats(cleaned_data[ticker], window=20)\n",
                "    plt.plot(roll_std, label=f\"{ticker} 20d Rolling Volatility\")\n",
                "plt.title(\"20-day Rolling Volatility (Standard Deviation)\", fontsize=14)\n",
                "plt.xlabel(\"Date\", fontsize=12)\n",
                "plt.ylabel(\"Volatility\", fontsize=12)\n",
                "plt.legend()\n",
                "plt.grid(True)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### D. Normality Diagnostics: Distribution & Box Plots\n",
                "We overlay return histograms with normal distribution curves and plot return boxplots to evaluate fat tails and outliers."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for ticker in tickers:\n",
                "    rets = returns_dict[ticker]\n",
                "    plt.figure(figsize=(10, 4))\n",
                "    sns.histplot(rets, kde=True, stat=\"density\", label=\"Historical Returns\", bins=50, alpha=0.6)\n",
                "    \n",
                "    # Normal curve fit overlay\n",
                "    mu, std = norm.fit(rets)\n",
                "    xmin, xmax = plt.xlim()\n",
                "    x = np.linspace(xmin, xmax, 100)\n",
                "    p = norm.pdf(x, mu, std)\n",
                "    plt.plot(x, p, 'r--', linewidth=2, label=f\"Normal Fit (mu={mu:.4f}, std={std:.4f})\")\n",
                "    \n",
                "    plt.title(f\"{ticker} Returns Distribution vs. Normal Curve\", fontsize=12)\n",
                "    plt.xlabel(\"Daily Return\")\n",
                "    plt.ylabel(\"Density\")\n",
                "    plt.legend()\n",
                "    plt.grid(True)\n",
                "    plt.show()\n",
                "    \n",
                "    plt.figure(figsize=(5, 3))\n",
                "    sns.boxplot(y=rets)\n",
                "    plt.title(f\"{ticker} Returns Boxplot (Outliers)\", fontsize=12)\n",
                "    plt.ylabel(\"Daily Return\")\n",
                "    plt.grid(True)\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### E. Autocorrelation (ACF / PACF) Analysis\n",
                "We construct ACF and PACF plots for daily returns to study correlation patterns before modeling."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for ticker in tickers:\n",
                "    rets = returns_dict[ticker]\n",
                "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))\n",
                "    plot_acf(rets, lags=40, ax=ax1, title=f\"{ticker} ACF (Returns)\")\n",
                "    plot_pacf(rets, lags=40, ax=ax2, title=f\"{ticker} PACF (Returns)\")\n",
                "    plt.tight_layout()\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### F. Time Series Decomposition (Trend, Seasonal, Residual)\n",
                "We decompose the adjusted closing prices to visualize trend and seasonal components over a 365d daily period."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for ticker in tickers:\n",
                "    prices = cleaned_data[ticker]['Adj Close']\n",
                "    prices_resampled = prices.asfreq('D').ffill()\n",
                "    result = seasonal_decompose(prices_resampled, model='additive', period=365)\n",
                "    \n",
                "    fig = result.plot()\n",
                "    fig.set_size_inches(12, 8)\n",
                "    plt.suptitle(f\"{ticker} Price Additive Decomposition (365d Period)\", y=1.02, fontsize=14)\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### G. Outlier Detection\n",
                "Identify dates where daily returns are greater than 3 standard deviations away from the mean."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for ticker in tickers:\n",
                "    outliers = detect_outliers(returns_dict[ticker], threshold=3.0)\n",
                "    print(f\"\\n{ticker} has {len(outliers)} outliers using a 3-standard-deviation threshold.\")\n",
                "    print(\"Top 5 most extreme outlier days:\")\n",
                "    print(outliers.abs().sort_values(ascending=False).head(5))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Seasonality and Trend Analysis (ADF Stationarity Test)\n",
                "\n",
                "We run the Augmented Dickey-Fuller (ADF) test on both closing prices and daily returns."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for ticker in tickers:\n",
                "    print(f\"\\n=== ADF Test for {ticker} ===\")\n",
                "    price_adf = run_adf_test(cleaned_data[ticker]['Adj Close'])\n",
                "    print(f\"Adj Close - ADF Stat: {price_adf['adf_stat']:.4f}, p-value: {price_adf['p_value']:.4e}\")\n",
                "    print(f\"  Is stationary? {price_adf['is_stationary']}\")\n",
                "    \n",
                "    ret_adf = run_adf_test(returns_dict[ticker])\n",
                "    print(f\"Daily Returns - ADF Stat: {ret_adf['adf_stat']:.4f}, p-value: {ret_adf['p_value']:.4e}\")\n",
                "    print(f\"  Is stationary? {ret_adf['is_stationary']}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Risk & Statistics Diagnostics"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for ticker in tickers:\n",
                "    var_95 = calculate_var(returns_dict[ticker], confidence_level=0.95)\n",
                "    var_99 = calculate_var(returns_dict[ticker], confidence_level=0.99)\n",
                "    cvar_95 = calculate_cvar(returns_dict[ticker], confidence_level=0.95)\n",
                "    cvar_99 = calculate_cvar(returns_dict[ticker], confidence_level=0.99)\n",
                "    max_dd = calculate_max_drawdown(cleaned_data[ticker]['Adj Close'])\n",
                "    sharpe = calculate_sharpe_ratio(returns_dict[ticker])\n",
                "    sortino = calculate_sortino_ratio(returns_dict[ticker])\n",
                "    skewness = calculate_skewness(returns_dict[ticker])\n",
                "    kurtosis = calculate_kurtosis(returns_dict[ticker])\n",
                "    jb_res = run_jarque_bera_test(returns_dict[ticker])\n",
                "    \n",
                "    print(f\"\\n--- Advanced Risk & Normality Metrics for {ticker} ---\")\n",
                "    print(f\"Value at Risk (95% Confidence): {var_95*100:.2f}%\")\n",
                "    print(f\"Value at Risk (99% Confidence): {var_99*100:.2f}%\")\n",
                "    print(f\"Conditional VaR (95% Confidence): {cvar_95*100:.2f}%\")\n",
                "    print(f\"Conditional VaR (99% Confidence): {cvar_99*100:.2f}%\")\n",
                "    print(f\"Maximum Historical Drawdown: {max_dd*100:.2f}%\")\n",
                "    print(f\"Annualized Sharpe Ratio: {sharpe:.4f}\")\n",
                "    print(f\"Annualized Sortino Ratio: {sortino:.4f}\")\n",
                "    print(f\"Skewness: {skewness:.4f}\")\n",
                "    print(f\"Excess Kurtosis: {kurtosis:.4f}\")\n",
                "    print(f\"Jarque-Bera Stat: {jb_res['jb_stat']:.2f}, p-value: {jb_res['p_value']:.4e}\")\n",
                "    print(f\"Is return normally distributed? {jb_res['is_normal']}\")"
            ]
        }
    ]
    
    create_notebook(os.path.join(notebooks_dir, "eda_and_preprocess.ipynb"), cells_eda)
    
    # ---------------------------------------------
    # Notebook 2: forecasting_models.ipynb
    # ---------------------------------------------
    cells_forecast = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Task 2 & 3: Build Time Series Forecasting Models and Predict Future Trends\n",
                "\n",
                "This notebook builds, tunes, and compares an **ARIMA/SARIMA** model and a **PyTorch LSTM** model to forecast Tesla's stock price. Then, we use the best-performing model to forecast future prices (6-12 months out) with confidence intervals."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "sys.path.append('../')\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "from statsmodels.tsa.statespace.sarimax import SARIMAX\n",
                "from src.data_loader import load_local_data\n",
                "from src.models import (\n",
                "    calculate_metrics, fit_auto_arima, prepare_lstm_data,\n",
                "    train_lstm, forecast_lstm_test, forecast_lstm_future\n",
                ")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Prepare Data for Modeling (Train-Test Split)\n",
                "We split the TSLA data chronologically. Train: 2015-2024, Test: 2025-2026."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "data_dict = load_local_data([\"TSLA\"], input_dir=\"../data/processed\")\n",
                "tsla_close = data_dict[\"TSLA\"]['Adj Close']\n",
                "\n",
                "train_split_date = \"2025-01-01\"\n",
                "train_close = tsla_close[:train_split_date]\n",
                "test_close = tsla_close[train_split_date:]\n",
                "\n",
                "print(f\"Train size: {len(train_close)}, Test size: {len(test_close)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Classical ARIMA/SARIMA Model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Find optimal order and fit\n",
                "arima_model = fit_auto_arima(train_close, m=1)\n",
                "arima_order = arima_model.order\n",
                "arima_seasonal_order = arima_model.seasonal_order\n",
                "\n",
                "# Fit statsmodels SARIMAX\n",
                "model_sm = SARIMAX(train_close, order=arima_order, seasonal_order=arima_seasonal_order)\n",
                "fitted_sm = model_sm.fit(disp=False)\n",
                "\n",
                "# Forecast on test period\n",
                "arima_test_preds = fitted_sm.forecast(steps=len(test_close))\n",
                "arima_test_preds.index = test_close.index"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Deep Learning LSTM Model (PyTorch)\n",
                "We frame this as a 60-day lag problem: predict today's price using the last 60 days of prices."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "window_size = 60\n",
                "X_train, y_train, scaler = prepare_lstm_data(train_close, window_size)\n",
                "lstm_model, device = train_lstm(X_train, y_train, epochs=20, batch_size=32, lr=0.001)\n",
                "\n",
                "# Prepare test dataset with overlap of last 60 days from training\n",
                "lstm_full_series = pd.concat([train_close[-window_size:], test_close])\n",
                "X_test, y_test, _ = prepare_lstm_data(lstm_full_series, window_size)\n",
                "\n",
                "lstm_test_preds = forecast_lstm_test(lstm_model, X_test, scaler, device)\n",
                "lstm_test_preds = pd.Series(lstm_test_preds, index=test_close.index)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Evaluate and Compare Models\n",
                "We calculate MAE, RMSE, and MAPE for both models on the test set."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "arima_metrics = calculate_metrics(test_close, arima_test_preds)\n",
                "lstm_metrics = calculate_metrics(test_close, lstm_test_preds)\n",
                "\n",
                "metrics_df = pd.DataFrame({\n",
                "    \"ARIMA\": arima_metrics,\n",
                "    \"LSTM\": lstm_metrics\n",
                "})\n",
                "print(metrics_df)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(14, 7))\n",
                "plt.plot(train_close[-150:], label=\"Train Actual (last 150 days)\", color='grey')\n",
                "plt.plot(test_close, label=\"Test Actual\", color='black')\n",
                "plt.plot(arima_test_preds, label=\"ARIMA Test Pred\", color='blue')\n",
                "plt.plot(lstm_test_preds, label=\"LSTM Test Pred\", color='orange')\n",
                "plt.title(\"Forecasting Performance Comparison on Test Set\", fontsize=14)\n",
                "plt.xlabel(\"Date\", fontsize=12)\n",
                "plt.ylabel(\"Price ($)\", fontsize=12)\n",
                "plt.legend()\n",
                "plt.grid(True)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Task 3: Forecast Future Market Trends (Next 6-12 Months)\n",
                "We fit the best-performing model on the full TSLA dataset and generate predictions for the next 6 months."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "forecast_steps = 126 # ~6 months\n",
                "\n",
                "# Fit ARIMA on full dataset\n",
                "model_sm_full = SARIMAX(tsla_close, order=arima_order, seasonal_order=arima_seasonal_order)\n",
                "fitted_sm_full = model_sm_full.fit(disp=False)\n",
                "\n",
                "arima_future_forecast = fitted_sm_full.get_forecast(steps=forecast_steps)\n",
                "arima_future_mean = arima_future_forecast.predicted_mean\n",
                "arima_future_ci = arima_future_forecast.conf_int(alpha=0.05)\n",
                "\n",
                "# Fit LSTM future (iterative rollout)\n",
                "last_sequence_scaled = scaler.transform(tsla_close[-window_size:].values.reshape(-1, 1))\n",
                "lstm_future_mean = forecast_lstm_future(lstm_model, last_sequence_scaled, forecast_steps, scaler, device)\n",
                "\n",
                "future_dates = pd.date_range(start=tsla_close.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')\n",
                "arima_future_mean.index = future_dates\n",
                "arima_future_ci.index = future_dates\n",
                "lstm_future_mean = pd.Series(lstm_future_mean, index=future_dates)\n",
                "\n",
                "# Plot\n",
                "plt.figure(figsize=(14, 7))\n",
                "plt.plot(tsla_close[-250:], label=\"Historical Actual (last year)\", color='black')\n",
                "plt.plot(arima_future_mean, label=\"ARIMA Future Forecast\", color='blue')\n",
                "plt.fill_between(future_dates, arima_future_ci.iloc[:, 0], arima_future_ci.iloc[:, 1], color='blue', alpha=0.15, label=\"ARIMA 95% Confidence Interval\")\n",
                "plt.plot(lstm_future_mean, label=\"LSTM Future Forecast\", color='orange')\n",
                "plt.title(\"Tesla (TSLA) Future Price Forecasting\", fontsize=14)\n",
                "plt.xlabel(\"Date\", fontsize=12)\n",
                "plt.ylabel(\"Price ($)\", fontsize=12)\n",
                "plt.legend()\n",
                "plt.grid(True)\n",
                "plt.show()"
            ]
        }
    ]
    
    create_notebook(os.path.join(notebooks_dir, "forecasting_models.ipynb"), cells_forecast)
    
    # ---------------------------------------------
    # Notebook 3: portfolio_optimization.ipynb
    # ---------------------------------------------
    cells_portfolio = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Task 4 & 5: Portfolio Optimization and Strategy Backtesting\n",
                "\n",
                "This notebook implements Modern Portfolio Theory (MPT) to optimize a portfolio of **TSLA**, **BND**, and **SPY** by combining forecasted returns with historical data. Then, we backtest this optimized strategy against a 60/40 benchmark."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "sys.path.append('../')\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from src.data_loader import load_local_data\n",
                "from src.eda_utils import calculate_daily_returns\n",
                "from src.portfolio import (\n",
                "    optimize_portfolio, generate_efficient_frontier, get_portfolio_metrics\n",
                ")\n",
                "from src.backtest import run_backtest, calculate_backtest_metrics"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Data and Prepare Returns\n",
                "We use the forecasted return of TSLA and historical returns of BND and SPY."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tickers = [\"TSLA\", \"BND\", \"SPY\"]\n",
                "data_dict = load_local_data(tickers, input_dir=\"../data/processed\")\n",
                "\n",
                "# Calculate daily historical returns\n",
                "returns_df = pd.DataFrame({\n",
                "    t: calculate_daily_returns(data_dict[t]) for t in tickers\n",
                "}).dropna()\n",
                "\n",
                "# Expected returns: For TSLA, load the forecasted returns from task 3.\n",
                "# Here we calculate the average forecasted daily return and annualize it.\n",
                "future_forecast = pd.read_csv(\"../data/processed/future_forecast.csv\", index_col=0)\n",
                "# Let's use the ARIMA future forecast mean\n",
                "tsla_forecast_returns = future_forecast[\"ARIMA_Mean\"].pct_change().dropna()\n",
                "tsla_expected_return = tsla_forecast_returns.mean() * 252\n",
                "\n",
                "expected_returns = pd.Series({\n",
                "    \"TSLA\": tsla_expected_return,\n",
                "    \"BND\": returns_df[\"BND\"].mean() * 252,\n",
                "    \"SPY\": returns_df[\"SPY\"].mean() * 252\n",
                "})\n",
                "print(\"Expected Annualized Returns:\")\n",
                "print(expected_returns)\n",
                "\n",
                "# Covariance Matrix (annualized)\n",
                "cov_matrix = returns_df.cov() * 252\n",
                "print(\"\\nCovariance Matrix (Annualized):\")\n",
                "print(cov_matrix)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Correlation and Covariance Heatmap"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(6, 5))\n",
                "sns.heatmap(returns_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=\".2f\")\n",
                "plt.title(\"Correlation Heatmap\", fontsize=12)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Generate Efficient Frontier\n",
                "We run simulations and solve for the optimal portfolios."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "weights_max_sharpe, weights_min_vol = optimize_portfolio(expected_returns, cov_matrix)\n",
                "print(\"Max Sharpe Portfolio Weights:\", weights_max_sharpe)\n",
                "print(\"Min Volatility Portfolio Weights:\", weights_min_vol)\n",
                "\n",
                "# Run simulations for plotting the frontier\n",
                "results, weights_record = generate_efficient_frontier(expected_returns, cov_matrix, num_portfolios=5000)\n",
                "\n",
                "# Retrieve optimized portfolios metrics\n",
                "w_ms = [weights_max_sharpe[t] for t in tickers]\n",
                "ret_ms, vol_ms, sharpe_ms = get_portfolio_metrics(w_ms, expected_returns, cov_matrix)\n",
                "\n",
                "w_mv = [weights_min_vol[t] for t in tickers]\n",
                "ret_mv, vol_mv, sharpe_mv = get_portfolio_metrics(w_mv, expected_returns, cov_matrix)\n",
                "\n",
                "plt.figure(figsize=(10, 6))\n",
                "sc = plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', s=10, alpha=0.3)\n",
                "plt.colorbar(sc, label='Sharpe Ratio')\n",
                "plt.scatter(vol_ms, ret_ms, color='red', marker='*', s=200, label='Max Sharpe Ratio')\n",
                "plt.scatter(vol_mv, ret_mv, color='blue', marker='X', s=200, label='Min Volatility')\n",
                "plt.title(\"Efficient Frontier\", fontsize=14)\n",
                "plt.xlabel(\"Annualized Volatility (Risk)\", fontsize=12)\n",
                "plt.ylabel(\"Expected Return\", fontsize=12)\n",
                "plt.legend()\n",
                "plt.grid(True)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Task 5: Backtesting\n",
                "We test our Max Sharpe Ratio strategy against a static benchmark of 60% SPY / 40% BND during the 2025 period (rebalanced monthly)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "backtest_returns = returns_df.loc[\"2025-01-01\":\"2026-01-01\"]\n",
                "\n",
                "# Optimized Strategy (Max Sharpe, Monthly Rebalanced)\n",
                "strat_value, _ = run_backtest(backtest_returns, weights_max_sharpe, rebalance_monthly=True)\n",
                "strat_metrics = calculate_backtest_metrics(strat_value)\n",
                "\n",
                "# Benchmark (60% SPY / 40% BND, Buy & Hold)\n",
                "benchmark_weights = {\"TSLA\": 0.0, \"SPY\": 0.6, \"BND\": 0.4}\n",
                "bench_value, _ = run_backtest(backtest_returns, benchmark_weights, rebalance_monthly=False)\n",
                "bench_metrics = calculate_backtest_metrics(bench_value)\n",
                "\n",
                "metrics_compare = pd.DataFrame({\n",
                "    \"Optimized Strategy\": strat_metrics,\n",
                "    \"60/40 Benchmark\": bench_metrics\n",
                "})\n",
                "print(metrics_compare)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(14, 7))\n",
                "plt.plot((strat_value / strat_value.iloc[0]) - 1, label=\"Optimized Strategy\", color='blue')\n",
                "plt.plot((bench_value / bench_value.iloc[0]) - 1, label=\"Benchmark (60/40 SPY/BND)\", color='grey', linestyle='--')\n",
                "plt.title(\"Backtesting Performance: Cumulative Returns Comparison (2025)\", fontsize=14)\n",
                "plt.xlabel(\"Date\", fontsize=12)\n",
                "plt.ylabel(\"Cumulative Return\", fontsize=12)\n",
                "plt.legend()\n",
                "plt.grid(True)\n",
                "plt.show()"
            ]
        }
    ]
    
    create_notebook(os.path.join(notebooks_dir, "portfolio_optimization.ipynb"), cells_portfolio)

if __name__ == "__main__":
    main()
