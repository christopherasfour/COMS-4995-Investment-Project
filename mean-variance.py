# import pandas as pd
# import numpy as np
# from scipy.optimize import minimize

# # Step 2: Load the historical price data for each stock from the CSV files
# stocks = ['AAPL', 'AMD', 'AMZN', 'CCJ', 'COST', 'GOOG', 'GS', 'JPM', 'LLY', 'META', 'MSFT', 'NEE', 'PFE', 'SAP', 'WMT']
# prices = pd.DataFrame()

# for stock in stocks:
#     filename = stock + '.csv'
#     data = pd.read_csv(filename)
#     prices[stock] = data['Close']

# # Step 3: Calculate the returns from the price data
# returns = prices.pct_change().dropna()

# # Step 4: Define the objective function for mean-variance optimization
# def portfolio_variance(weights, returns):
#     cov_matrix = returns.cov()
#     portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
#     return portfolio_var

# # Step 5: Define the constraints for the optimization problem
# def weight_constraint(weights):
#     return np.sum(weights) - 1.0

# # Step 6: Set the initial guess for the weights and specify constraints
# num_assets = len(stocks)
# weights_guess = np.ones(num_assets) / num_assets
# constraints = ({'type': 'eq', 'fun': weight_constraint})

# # Step 7: Run the optimization using the minimize function
# opt_results = minimize(portfolio_variance, weights_guess, args=(returns,), method='SLSQP', constraints=constraints)

# # Step 8: Extract the optimized weights and calculate other portfolio statistics
# optimized_weights = opt_results.x
# portfolio_return = np.sum(returns.mean() * optimized_weights)
# portfolio_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(returns.cov(), optimized_weights)))

# # Step 9: Print the optimized weights and portfolio statistics
# print("Optimized Weights:")
# for stock, weight in zip(stocks, optimized_weights):
#     print(stock + ":", weight)

# print("\nPortfolio Return:", portfolio_return)
# print("Portfolio Volatility:", portfolio_volatility)
# risk_free_rate = 0.0  # Adjust according to your risk-free rate assumption
# portfolio_sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

# print("Sharpe Ratio:", portfolio_sharpe_ratio)

# import pandas as pd
# import numpy as np
# from scipy.optimize import minimize

# # Define the list of stocks
# stocks = ['AAPL', 'AMD', 'AMZN', 'CCJ', 'COST', 'GOOG', 'GS', 'JPM', 'LLY', 'META', 'MSFT', 'NEE', 'PFE', 'SAP', 'WMT']

# # Read historical data from CSV files
# data = pd.DataFrame()
# for stock in stocks:
#     filename = stock + '.csv'
#     stock_data = pd.read_csv(filename)
#     data[stock] = stock_data['Close']

# # Calculate returns
# returns = data.pct_change().dropna()

# # Define the objective function to minimize
# def objective(weights):
#     port_returns = np.dot(returns.mean(), weights)
#     port_variance = np.dot(weights, np.dot(returns.cov(), weights))
#     return port_variance

# # Define the constraints
# def constraint(weights):
#     return np.sum(weights) - 1

# # Set the initial guess for the weights
# weights_guess = np.ones(len(stocks)) / len(stocks)

# # Define the bounds for the weights (between 0 and 1)
# bounds = [(0, 1)] * len(stocks)

# # Define the optimization problem
# problem = {'type': 'eq', 'fun': constraint}

# # Run the optimization
# result = minimize(objective, weights_guess, method='SLSQP', bounds=bounds, constraints=problem)

# # Print the optimized portfolio weights
# print('Optimized Portfolio Weights:')
# for i, stock in enumerate(stocks):
#     print(stock + ':', result.x[i])

import pandas as pd
import numpy as np
from scipy.optimize import minimize

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

# Perform mean-variance optimization
result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': constraint})

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
