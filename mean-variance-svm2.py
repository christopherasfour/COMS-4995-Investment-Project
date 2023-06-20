import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVR

# List of stock symbols
stocks = ['AAPL', 'AMD', 'AMZN', 'CCJ', 'COST', 'GOOG', 'GS', 'JPM', 'LLY', 'META', 'MSFT', 'NEE', 'PFE', 'SAP', 'WMT']

# Load stock data from CSV files
data = {}
for stock in stocks:
    filename = stock + '.csv'
    data[stock] = pd.read_csv(filename)

# Combine all stock data into a single DataFrame
prices_df = pd.DataFrame()
for stock in stocks:
    prices_df[stock] = data[stock]['Adj Close']

# Calculate daily returns
returns_df = prices_df.pct_change().dropna()

# Define the objective function for mean-variance optimization
def objective(weights):
    portfolio_return = np.sum(returns_df.mean() * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov(), weights)))
    return -portfolio_return / portfolio_volatility  # Minimize the negative Sharpe ratio

# Define the constraint that the sum of weights equals 1
def constraint(weights):
    return np.sum(weights) - 1

# Define the weight bounds (between 0 and 1)
bounds = [(0, 1)] * len(stocks)

# Define the initial guess for weights
initial_weights = np.ones(len(stocks)) / len(stocks)

# Train SVR models for each stock
regressors = {}
for stock in stocks:
    regressor = SVR(kernel='rbf')
    regressor.fit(returns_df, returns_df[stock])
    regressors[stock] = regressor

# Define the objective function for SVR optimization
def svr_objective(weights):
    portfolio_return = np.sum([regressors[stock].predict(returns_df) * weight for stock, weight in zip(stocks, weights)], axis=0)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov(), weights)))
    return -np.mean(portfolio_return) / portfolio_volatility  # Minimize the negative Sharpe ratio

# Perform SVR optimization
result = minimize(svr_objective, initial_weights, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': constraint})

# Extract the optimized weights
optimized_weights = result.x

# Calculate portfolio statistics
portfolio_return = np.sum(returns_df.mean() * optimized_weights)
portfolio_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(returns_df.cov(), optimized_weights)))
sharpe_ratio = portfolio_return / portfolio_volatility

# Print the results
print("Optimized Weights:")
for stock, weight in zip(stocks, optimized_weights):
    print(f"{stock}: {weight:.4f}")
print()
print("Portfolio Return:", portfolio_return)
print("Portfolio Volatility:", portfolio_volatility)
print("Sharpe Ratio:", sharpe_ratio)
