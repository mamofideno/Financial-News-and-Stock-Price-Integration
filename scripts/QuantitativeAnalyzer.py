import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt

class QuantitativeAnalysis:
    def __init__(self, df, price_column='Close'):
        """
        Initialize the QuantitativeAnalysis class.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing stock data with at least 'Close' prices.
        price_column (str): The name of the column containing the closing prices. Default is 'Close'.
        """
        self.df = df
        self.price_column = price_column

    def calculate_moving_averages(self, short_window=50, long_window=200):
        """
        Calculate short and long period moving averages.
        
        Parameters:
        short_window (int): The period for the short-term moving average (e.g., 50 days).
        long_window (int): The period for the long-term moving average (e.g., 200 days).
        """
        self.df['SMA_' + str(short_window)] = talib.SMA(self.df[self.price_column], timeperiod=short_window)
        self.df['SMA_' + str(long_window)] = talib.SMA(self.df[self.price_column], timeperiod=long_window)
        self.df['EMA_' + str(short_window)] = talib.EMA(self.df[self.price_column], timeperiod=short_window)
    
    def calculate_rsi(self, period=14):
        """
        Calculate the Relative Strength Index (RSI).
        
        Parameters:
        period (int): The period for calculating RSI. Default is 14 days.
        """
        self.df['RSI'] = talib.RSI(self.df[self.price_column], timeperiod=period)
    
    def calculate_bollinger_bands(self, period=20, nbdevup=2, nbdevdn=2):
        """
        Calculate Bollinger Bands.
        
        Parameters:
        period (int): The period for the moving average. Default is 20 days.
        nbdevup (int): Number of standard deviations for the upper band. Default is 2.
        nbdevdn (int): Number of standard deviations for the lower band. Default is 2.
        """
        self.df['upper_band'], self.df['middle_band'], self.df['lower_band'] = talib.BBANDS(
            self.df[self.price_column], timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn
        )
    
    def calculate_macd(self, fastperiod=12, slowperiod=26, signalperiod=9):
        """
        Calculate the Moving Average Convergence Divergence (MACD).
        
        Parameters:
        fastperiod (int): The period for the fast EMA. Default is 12 days.
        slowperiod (int): The period for the slow EMA. Default is 26 days.
        signalperiod (int): The period for the signal line. Default is 9 days.
        """
        self.df['MACD'], self.df['MACD_signal'], self.df['MACD_hist'] = talib.MACD(
            self.df[self.price_column], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod
        )
    
    def calculate_volatility(self, window=252):
        """
        Calculate annualized volatility.
        
        Parameters:
        window (int): The number of trading days to annualize. Default is 252 days.
        
        Returns:
        float: Annualized volatility.
        """
        daily_returns = self.df[self.price_column].pct_change()
        volatility = daily_returns.rolling(window=window).std() * np.sqrt(window)
        self.df['Volatility'] = volatility
        return volatility.iloc[-1]

    def calculate_sharpe_ratio(self, risk_free_rate=0.01, window=252):
        """
        Calculate the Sharpe Ratio.
        
        Parameters:
        risk_free_rate (float): The risk-free rate for Sharpe Ratio calculation. Default is 1%.
        window (int): The number of trading days to annualize. Default is 252 days.
        
        Returns:
        float: Sharpe Ratio.
        """
        daily_returns = self.df[self.price_column].pct_change()
        excess_returns = daily_returns - (risk_free_rate / window)
        sharpe_ratio = np.sqrt(window) * excess_returns.mean() / excess_returns.std()
        self.df['Sharpe_Ratio'] = sharpe_ratio
        return sharpe_ratio
    
    def plot_technical_indicators(self, ticker):
        """
        Plot the calculated technical indicators.
        
        Parameters:
        ticker (str): The stock ticker for the plots' title.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(self.df[self.price_column], label=f'{ticker} Close Price', color='blue')
        plt.plot(self.df['SMA_50'], label='50-day SMA', color='red')
        plt.plot(self.df['SMA_200'], label='200-day SMA', color='green')
        plt.plot(self.df['EMA_50'], label='50-day EMA', color='orange')
        plt.title(f'{ticker} - Technical Analysis')
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(14, 7))
        plt.plot(self.df['RSI'], label=f'{ticker} RSI', color='purple')
        plt.axhline(70, color='red', linestyle='--')
        plt.axhline(30, color='green', linestyle='--')
        plt.title(f'{ticker} - Relative Strength Index')
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(14, 7))
        plt.plot(self.df[self.price_column], label=f'{ticker} Close Price', color='blue')
        plt.plot(self.df['upper_band'], label='Upper Bollinger Band', color='red')
        plt.plot(self.df['middle_band'], label='Middle Bollinger Band', color='green')
        plt.plot(self.df['lower_band'], label='Lower Bollinger Band', color='red')
        plt.title(f'{ticker} - Bollinger Bands')
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(14, 7))
        plt.plot(self.df['MACD'], label='MACD', color='blue')
        plt.plot(self.df['MACD_signal'], label='Signal Line', color='red')
        plt.bar(self.df.index, self.df['MACD_hist'], label='MACD Histogram', color='gray')
        plt.title(f'{ticker} - MACD')
        plt.legend()
        plt.show()

