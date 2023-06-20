import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization

# List of stock symbols
stocks = ['AAPL', 'AMD', 'AMZN', 'CCJ', 'COST', 'GOOG', 'GS', 'JPM', 'LLY', 'META', 'MSFT', 'NEE', 'PFE', 'SAP', 'WMT']

# Load stock data from CSV files
data = {}
for stock in stocks:
    filename = stock + '.csv'
    data[stock] = pd.read_csv(filename)

data['SPY'] = pd.read_csv('SPY.csv')

# Combine all stock data into a single DataFrame
prices_df = pd.DataFrame()
for stock in stocks:
    prices_df[stock] = data[stock]['Adj Close']

# Calculate daily returns
returns_df = prices_df.pct_change().dropna()

# Calculate the market returns (you can replace 'SPY' with any suitable market index)
market_returns = data['SPY']['Adj Close'].pct_change().dropna()

# Calculate stock betas using index model
betas = {}
for stock in stocks:
    cov_matrix = np.cov(returns_df[stock], market_returns)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    betas[stock] = beta

# Define the objective function for Bayesian optimization
def objective(**weights):
    weights = np.array(list(weights.values()))
    portfolio_return = np.sum(returns_df.mean() * weights)
    portfolio_beta = np.sum([betas[stock] * weight for stock, weight in zip(stocks, weights)])
    return -portfolio_return / portfolio_beta  # Minimize the negative risk-adjusted return

# Define the bounds for Bayesian optimization (between 0 and 1 for each stock)
bounds = {stock: (0, 1) for stock in stocks}

# Perform Bayesian optimization
optimizer = BayesianOptimization(f=objective, pbounds=bounds)
optimizer.maximize(init_points=10, n_iter=50)

# Extract the optimized weights
optimized_weights = np.array(list(optimizer.max['params'].values()))

# Calculate portfolio statistics
portfolio_return = np.sum(returns_df.mean() * optimized_weights)
portfolio_beta = np.sum([betas[stock] * weight for stock, weight in zip(stocks, optimized_weights)])
risk_adjusted_return = portfolio_return / portfolio_beta

# Print the results
print("Optimized Weights:")
for stock, weight in zip(stocks, optimized_weights):
    print(f"{stock}: {weight:.4f}")
print()
print("Portfolio Return:", portfolio_return)
print("Portfolio Beta:", portfolio_beta)
print("Risk-Adjusted Return:", risk_adjusted_return)
