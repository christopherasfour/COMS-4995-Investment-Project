import yfinance as yf
from pypfopt import expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt.efficient_frontier import EfficientFrontier
from sklearn.linear_model import LogisticRegression
import numpy as np

# Define the list of stocks
stocks = ['AAPL', 'AMD', 'AMZN', 'CCJ', 'COST', 'GOOG', 'GS', 'JPM', 'LLY', 'META', 'MSFT', 'NEE', 'PFE', 'SAP', 'WMT']

# Fetch historical stock data
data = yf.download(stocks, start='2018-01-01', end='2023-06-23')['Adj Close']

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(data)
S = risk_models.sample_cov(data)

# Create an instance of the Logistic Regression model
log_reg = LogisticRegression(solver='lbfgs')

# Fit the Logistic Regression model
log_reg.fit(data, np.ones(data.shape[0]))  # Dummy target variable

# Predict probabilities of positive class (to be used as expected returns)
expected_returns = log_reg.predict_proba(data)[:, 1]

# Create the Efficient Frontier object
ef = EfficientFrontier(expected_returns, S)

# Optimize for maximum Sharpe ratio
weights = ef.max_sharpe()

# Get the discrete allocation of assets
da = DiscreteAllocation(weights, data.iloc[-1], total_portfolio_value=10000)

allocation, leftover = da.lp_portfolio()

# Print the allocation
print(allocation)
print("Funds remaining: ${:.2f}".format(leftover))
