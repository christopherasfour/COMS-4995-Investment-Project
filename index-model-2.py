import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation

tickers = ['AAPL', 'AMD', 'AMZN', 'CCJ', 'COST', 'GOOG', 'GS', 'JPM', 'LLY', 'META', 'MSFT', 'NEE', 'PFE', 'SAP', 'WMT', "^GSPC"]

data = yf.download(tickers, start="2010-01-01", end="2023-06-24")

adj_close = data["Adj Close"]
log_returns = np.log(adj_close / adj_close.shift(1)).dropna()

X = log_returns["^GSPC"].values.reshape(-1, 1)
y = np.zeros((log_returns.shape[0], len(tickers)))

models = {}
for i, ticker in enumerate(tickers):
    y[:, i] = (log_returns[ticker] > log_returns["^GSPC"]).astype(int)
    model = LogisticRegression(random_state=0)
    model.fit(X, y[:, i])
    models[ticker] = model

mu = expected_returns.mean_historical_return(adj_close)
Sigma = risk_models.sample_cov(adj_close)

ef = EfficientFrontier(mu, Sigma)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)

latest_prices = data["Adj Close"].iloc[-1]
da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=10000)
allocation, leftover = da.lp_portfolio()

print("Discrete Allocation:", allocation)
print("Funds Remaining: ${:.2f}".format(leftover))
