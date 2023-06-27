import yfinance as yf
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from sklearn.linear_model import LinearRegression

# Define the list of stock tickers
tickers = ['AAPL', 'AMD', 'AMZN', 'CCJ', 'COST', 'GOOG', 'GS', 'JPM', 'LLY', 'META', 'MSFT', 'NEE', 'PFE', 'SAP', 'WMT']

# Download historical price data from Yahoo Finance using yfinance
prices = yf.download(tickers, period="5y")["Adj Close"]
returns = prices.pct_change().dropna()

# Use machine learning to estimate expected returns
model = LinearRegression()
expected_returns_ml = pd.Series(index=returns.columns)
for ticker in returns.columns:
    X = returns.drop(columns=[ticker]).values
    y = returns[ticker].values
    model.fit(X, y)
    expected_returns_ml[ticker] = model.predict([X[-1]])

# Use machine learning to estimate covariance matrix
covariance_ml = returns.cov()

# Create the Efficient Frontier object
ef = EfficientFrontier(expected_returns_ml, covariance_ml, weight_bounds=(0.05, 0.1))

# Optimize for minimum risk using Efficient Frontier
weights = ef.min_volatility()
weights = ef.clean_weights()

# Print the optimized portfolio weights and performance
print("Efficient Frontier Optimization:")
print(weights)
print(ef.portfolio_performance(verbose=True))
