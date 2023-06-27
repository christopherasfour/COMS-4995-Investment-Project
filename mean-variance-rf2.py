import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from sklearn.impute import SimpleImputer


# Define the list of stock tickers
tickers = ['AAPL', 'AMD', 'AMZN', 'CCJ', 'COST', 'GOOG', 'GS', 'JPM', 'LLY', 'META', 'MSFT', 'NEE', 'PFE', 'SAP', 'WMT']

# Download historical price data from Yahoo Finance using yfinance and calculate daily returns
prices = yf.download(tickers, period="max")["Adj Close"]
returns = prices.pct_change().dropna()

# Create and fit the Random Forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(returns.iloc[:-1], returns.shift(-1).iloc[:-1])

# Predict the next day's returns
predicted_returns = rf.predict(returns.iloc[-1].values.reshape(1, -1))
predicted_returns = pd.Series(predicted_returns[0], index=returns.columns)

# def calculate_expected_returns(prices):
#     # Handle missing values
#     imputer = SimpleImputer(strategy='mean')
#     prices_filled = pd.DataFrame(imputer.fit_transform(prices), index=prices.index, columns=prices.columns)

#     # Initialize a Random Forest Regressor
#     regressor = RandomForestRegressor(n_estimators=100, random_state=0)

#     # Prepare the training data
#     X = prices_filled.iloc[:-1].values  # Historical prices except the last row
#     y = prices_filled.iloc[1:].values   # Next day's prices

#     # Fit the regressor to the training data
#     regressor.fit(X, y)

#     # Use the regressor to predict the next day's prices
#     predicted_prices = regressor.predict(prices_filled.iloc[-1:].values)

#     # Calculate the expected returns from the predicted prices
#     expected_returns = pd.Series(predicted_prices[0] / prices_filled.iloc[-1].values - 1, index=prices_filled.columns)

#     return expected_returns



# Calculate expected returns and sample covariance matrix
expected_returns = predicted_returns
covariance = risk_models.sample_cov(prices)

# Create the Efficient Frontier object
ef = EfficientFrontier(expected_returns, covariance, weight_bounds=(0.05, 0.1))

# Optimize for minimum risk
weights = ef.min_volatility()
weights = ef.clean_weights()

# Print the optimized portfolio weights
print(weights)
print(ef.portfolio_performance(verbose=True))

from pypfopt import DiscreteAllocation

latest_prices = prices.iloc[-1]  # prices as of the day you are allocating
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=800000, short_ratio=0.0)
alloc, leftover = da.greedy_portfolio() #instead of lp_portfolio()
print(f"Discrete allocation performed with ${leftover:.2f} leftover")
print(alloc)


