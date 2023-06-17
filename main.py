import pandas as pd
import numpy as np
from scipy.optimize import minimize

tickers = ['AAPL', 'AMD', 'AMZN', 'CCJ', 'COST', 'GOOG', 'GS', 'JPM', 'LLY', 'META', 'MSFT', 'NEE', 'PFE', 'SAP', 'WMT']
stocks = {}
for ticker in tickers:
    filename = f"{ticker}.csv"
    data = pd.read_csv(filename)
    stocks[f"{ticker}"] = data

returns = {}
for stock, data in stocks.items():
    returns[stock] = data['Close'].pct_change().dropna()

returns_df = pd.DataFrame(returns)
expected_returns = returns_df.mean()
print(expected_returns)
cov_matrix = returns_df.cov()
print(cov_matrix)

def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def portfolio_return(weights, expected_returns):
    return np.dot(weights, expected_returns)

def objective_function(weights, expected_returns, cov_matrix, risk_aversion):
    portfolio_ret = portfolio_return(weights, expected_returns)
    portfolio_var = portfolio_variance(weights, cov_matrix)
    # return portfolio_var - risk_aversion * portfolio_ret
    sharpe_ratio = (portfolio_ret) / np.sqrt(portfolio_var)
    return -sharpe_ratio


# # Function to calculate portfolio returns and volatility
# def calculate_portfolio_perf(weights, returns):
#     portfolio_return = np.sum(np.mean(returns[stock] * weights[i]) * 252 for i, stock in enumerate(stocks))
#     # Calculate portfolio volatility using individual stock returns and covariances
#     # Create a list of individual stock returns
#     stock_returns = [returns[stock] for stock in stocks]

#     # Calculate the portfolio volatility using individual stock returns and covariances
#     portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(np.cov(stock_returns) * 252, weights)))

#     # portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns[stock]) * 252, weights[i]) for i, stock in enumerate(stocks)))
#     return portfolio_return, portfolio_volatility

# # Function to calculate negative Sharpe ratio (to be minimized)
# def negative_sharpe_ratio(weights, returns):
#     portfolio_return, portfolio_volatility = calculate_portfolio_perf(weights, returns)
#     sharpe_ratio = (portfolio_return) / portfolio_volatility
#     return -sharpe_ratio

# # Set initial weights
# weights = np.array([1/len(stocks)] * len(stocks))

# # Set optimization constraints
# constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
# # Set optimization bounds (0 <= weight <= 1)
# bounds = [(-1, 1)] * len(stocks)

# # Perform mean-variance portfolio optimization
# result = minimize(negative_sharpe_ratio, weights, args=(returns),
#                   method='SLSQP', bounds=bounds, constraints=constraints)

# # Get optimal weights
# optimal_weights = result.x

# # Print optimal weights for each stock
# for i, stock in enumerate(stocks):
#     print(f"Stock: {stock}, Allocation: {optimal_weights[i] * 100}%")

# # Calculate portfolio performance with optimal weights
# portfolio_return, portfolio_volatility = calculate_portfolio_perf(optimal_weights, returns)
# sharpe_ratio = (portfolio_return) / portfolio_volatility

# # Print portfolio performance metrics
# print(f"\nPortfolio Return: {portfolio_return * 100}%")
# print(f"Portfolio Volatility: {portfolio_volatility * 100}%")
# print(f"Sharpe Ratio: {sharpe_ratio}")

# # Set optimization constraints
constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
# # Set optimization bounds (0 <= weight <= 1)
bounds = [(-1, 1)] * len(stocks)


risk_aversion = 2
initial_weights = np.ones(len(stocks)) / len(stocks)
result = minimize(objective_function, initial_weights, args=(expected_returns, cov_matrix, risk_aversion),
                  method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x
print(optimal_weights)
print(sum(optimal_weights))

portfolio_balance = 1000000  # $1 million
allocations = portfolio_balance * optimal_weights

for stock, allocation in zip(stocks.keys(), allocations):
    print(f"Stock: {stock}, Allocation: {allocation}")
