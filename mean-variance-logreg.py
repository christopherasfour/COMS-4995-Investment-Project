import yfinance as yf
import numpy as np
from pypfopt import expected_returns, risk_models, discrete_allocation
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from sklearn.linear_model import LogisticRegression
from pypfopt import DiscreteAllocation


tickers = ['AAPL', 'AMD', 'AMZN', 'CCJ', 'COST', 'GOOG', 'GS', 'JPM', 'LLY', 'META', 'MSFT', 'NEE', 'PFE', 'SAP', 'WMT']

# Download historical stock data using yfinance
data = yf.download(tickers, period="max")
prices = data['Adj Close'].dropna(axis=1)  # Adjusted close prices
returns = prices.pct_change().dropna()  # Calculate returns

# Split data into features (X) and target (y)
X = returns.iloc[:-1]  # Use all but the last row as features
y = (returns.iloc[1:] > 0).any(axis=1)  # Target: True if any stock had positive return

# Train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Get predicted probabilities for the next time period
last_returns = returns.iloc[-1]
next_probabilities = model.predict_proba(last_returns.values.reshape(1, -1))

# Define expected returns using predicted probabilities
positive_probabilities = next_probabilities[:, 1]
expected_returns = last_returns * positive_probabilities


cov_matrix = returns.cov()

# Set up mean-variance optimization problem
ef = EfficientFrontier(expected_returns, cov_matrix)
ef.add_objective(objective_functions.L2_reg, gamma=0.1)  # Optional: L2 regularization
weights = ef.min_volatility()  # Maximize the Sharpe ratio

# Get discrete allocation of the portfolio based on available funds
available_funds = 1000000  # Example: $1,000,000
da = DiscreteAllocation(weights, last_returns, total_portfolio_value=available_funds)
allocation, leftover = da.greedy_portfolio()

# Print the portfolio allocation
print("Portfolio allocation:")
for ticker, shares in allocation.items():
    if shares > 0:
        print(f"{ticker}: {shares} shares")
