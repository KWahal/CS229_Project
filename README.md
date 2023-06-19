# CS229_Project
Final project for [CS 229: Machine Learning](https://cs229.stanford.edu/), Spring 2023.

Time-series Forecasting for Treasury Auctions. Karsen Wahal, Charlie Li, Soham Konar.

#### Abstract
The United States Treasury holds a variant of Dutch auctions to sell treasury bills (Tbills), special short-term bills issued when the government needs money for a short period of time. However, treasury bill auction high rates are notoriously difficult to predict given the complexity of financial markets. In this paper, we apply machine learning methods (linear regression, polynomial regression, ARIMA,
RNN-LSTM, CNN, and NeuralProphet) to predict treasury auction high rates based on historical data, economic sentiment, the type of treasury bill, and the investor class allocation. Additionally, we experiment with techniques such as cross-validation and resampling to better fine-tune our models. We find that RNN-LSTM models outperform the other variants and, in general, deep learning models capture the trends in the data more accurately than linear models.
