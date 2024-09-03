import pandas as pd
import numpy as np
import pynance as pn
from textblob import TextBlob
import matplotlib.pyplot as plt

class MultiTickerNewsStockCorrelation:
    def __init__(self, tickers, news_data):
        """
        Initializes the MultiTickerNewsStockCorrelation class.

        :param tickers: A list of stock ticker symbols (e.g., ['AAPL', 'GOOGL', 'MSFT']).
        :param news_data: A DataFrame with news headlines and their corresponding dates.
                          The DataFrame should have two columns: 'Date' and 'Headline'.
        """
        self.tickers = tickers
        self.news_data = news_data
        self.stock_data_dict = {}
        self.weekly_sentiment_scores = None
        self.weekly_stock_data_dict = {}

    def fetch_stock_data(self, start_date, end_date):
        """
        Fetches historical stock price data for each ticker.

        :param start_date: Start date for fetching data in 'YYYY-MM-DD' format.
        :param end_date: End date for fetching data in 'YYYY-MM-DD' format.
        """
        for ticker in self.tickers:
            stock_data = pn.data.get(ticker, start=start_date, end=end_date)
            if 'Adj Close' in stock_data.columns:
                stock_data['Adj Close'] = stock_data['Adj Close'].pct_change()
                # df['Adj_Close'] = df['Adj Close']
            elif 'Close' in stock_data.columns:
                stock_data['Adj Close'] = stock_data['Close'].pct_change()
            weekly_stock_data = stock_data['Adj Close'].resample('W').last().pct_change()
            self.stock_data_dict[ticker] = stock_data
            self.weekly_stock_data_dict[ticker] = weekly_stock_data

    def analyze_sentiment(self):
        """
        Analyzes the sentiment of the news headlines using TextBlob and aggregates it weekly.
        """
        def get_sentiment(text):
            return TextBlob(text).sentiment.polarity

        self.news_data['Sentiment'] = self.news_data['headline'].apply(get_sentiment)
        self.news_data['Date'] = pd.to_datetime(self.news_data['date'], dayfirst=False, errors='coerce')
        self.news_data.set_index('Date', inplace=True)
        # self.news_data['Date'] = pd.to_datetime(self.news_data['date'])
        # self.news_data.set_index('Date', inplace=True)
        
        # Resample sentiment scores to weekly frequency by averaging
        self.weekly_sentiment_scores = self.news_data['Sentiment'].resample('W').mean()

    def plot_correlation(self):
        """
        Plots the weekly sentiment scores and weekly stock price changes for all tickers on the same graph.
        """
        plt.figure(figsize=(14, 10))
        
        for ticker in self.tickers:
            plt.subplot(2, 1, 1)
            plt.plot(self.weekly_stock_data_dict[ticker].index, self.weekly_stock_data_dict[ticker], label=f'{ticker} Price Change')
        
        plt.title('Weekly Stock Price Change')
        plt.xlabel('Date')
        plt.ylabel('Weekly Price Change')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.weekly_sentiment_scores.index, self.weekly_sentiment_scores, color='orange', label='Sentiment Score')
        
        plt.title('Weekly News Sentiment Score')
        plt.xlabel('Date')
        plt.ylabel('Weekly Sentiment Score')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def run_analysis(self, start_date, end_date):
        """
        Runs the complete weekly analysis pipeline for all tickers.

        :param start_date: Start date for fetching data in 'YYYY-MM-DD' format.
        :param end_date: End date for fetching data in 'YYYY-MM-DD' format.
        """
        self.fetch_stock_data(start_date, end_date)
        self.analyze_sentiment()
        self.plot_correlation()

# Example usage:

# Load the news data from a CSV file
news_data = pd.read_csv('./Data/raw_analyst_ratings.csv')

# Initialize the correlation analysis class
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NFLX', 'META']
correlation_analysis = MultiTickerNewsStockCorrelation(tickers, news_data)

# Run the analysis
correlation_analysis.run_analysis('2023-07-01', '2023-08-31')
