import yfinance as yf
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Define the list of stocks
stocks = ['AAPL', 'AMD', 'AMZN', 'CCJ', 'COST', 'GOOG', 'GS', 'JPM', 'LLY', 'META', 'MSFT', 'NEE', 'PFE', 'SAP', 'WMT']

# Download historical stock data using yfinance
data = yf.download(stocks, start='2020-01-01', end='2023-01-01')['Adj Close']

# Prepare features and target for SVM
features = data.dropna().values[:-1]  # Use all features except the last row
target = data.dropna().values[1:]  # Predict the next day's prices

# Scale features using StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Train the SVM model
svm = SVR()
svm.fit(scaled_features, target)

# Predict the next day's prices
latest_features = scaler.transform(features[-1].reshape(1, -1))
predicted_prices = svm.predict(latest_features)[0]

# Create a DataFrame of predicted prices
predicted_prices_df = pd.DataFrame(predicted_prices, index=data.columns, columns=['Predicted Price'])

# Calculate expected returns and sample covariance matrix
mu = expected_returns.mean_historical_return(predicted_prices_df)
cov = risk_models.sample_cov(predicted_prices_df)

# Perform mean-variance portfolio optimization with EfficientFrontier
ef = EfficientFrontier(mu, cov)
weights = ef.max_sharpe()

# Get the optimal portfolio weights
cleaned_weights = ef.clean_weights()
print(cleaned_weights)

# Calculate portfolio performance metrics
portfolio_return = ef.portfolio_performance(verbose=True)
