import pandas as pd
import numpy as np
import pynance as pn
from textblob import TextBlob
import matplotlib.pyplot as plt

class NewsStockCorrelation:
    def __init__(self, ticker, news_data):
        """
        Initializes the NewsStockCorrelation class.

        :param ticker: The stock ticker symbol (e.g., 'AAPL').
        :param news_data: A DataFrame with news headlines and their corresponding dates.
                          The DataFrame should have two columns: 'Date' and 'Headline'.
        """
        self.ticker = ticker
        self.news_data = news_data
        self.stock_data = None
        self.sentiment_scores = None

    def fetch_stock_data(self, start_date, end_date):
        """
        Fetches historical stock price data.

        :param start_date: Start date for fetching data in 'YYYY-MM-DD' format.
        :param end_date: End date for fetching data in 'YYYY-MM-DD' format.
        """
        self.stock_data = pn.data.get(self.ticker, start=start_date, end=end_date)
        if 'Adj Close' in self.stock_data.columns:
                self.stock_data['Price_Change'] = self.stock_data['Adj Close'].pct_change()
                # df['Adj_Close'] = df['Adj Close']
        elif 'Close' in self.stock_data.columns:
                self.stock_data['Price_Change'] = self.stock_data['Close'].pct_change()
                # df['Adj_Close'] = df['Close']

    def analyze_sentiment(self):
        """
        Analyzes the sentiment of the news headlines using TextBlob.
        """
        def get_sentiment(text):
            return TextBlob(text).sentiment.polarity

        self.news_data['Sentiment'] = self.news_data['headline'].apply(get_sentiment)
        self.sentiment_scores = self.news_data.groupby('date')['Sentiment'].mean()

    def calculate_correlation(self):
        """
        Calculates the correlation between news sentiment and stock price movements.
        """
        # Merge sentiment scores with stock data
        combined_data = pd.merge(self.stock_data, self.sentiment_scores, left_index=True, right_index=True, how='inner')

        # Calculate correlation
        correlation = combined_data['Price_Change'].corr(combined_data['Sentiment'])
        return correlation

    def plot_correlation(self):
        """
        Plots the sentiment scores and stock price changes on the same graph.
        """
        plt.figure(figsize=(14, 7))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.stock_data.index, self.stock_data['Price_Change'], label='Price Change')
        plt.title(f'{self.ticker} Stock Price Change')
        plt.xlabel('Date')
        plt.ylabel('Price Change')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.sentiment_scores.index, self.sentiment_scores, color='orange', label='Sentiment Score')
        plt.title(f'{self.ticker} News Sentiment Score')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def run_analysis(self, start_date, end_date):
        """
        Runs the complete analysis pipeline.

        :param start_date: Start date for fetching data in 'YYYY-MM-DD' format.
        :param end_date: End date for fetching data in 'YYYY-MM-DD' format.
        :return: Correlation coefficient between news sentiment and stock price movements.
        """
        self.fetch_stock_data(start_date, end_date)
        self.analyze_sentiment()
        correlation = self.calculate_correlation()
        self.plot_correlation()
        return correlation

# Example usage:

# Sample news data
news_data = pd.read_csv('./Data/raw_analyst_ratings.csv')

# Initialize the correlation analysis class
ticker = 'AAPL'
correlation_analysis = NewsStockCorrelation(ticker, news_data)

# Run the analysis
correlation_coefficient = correlation_analysis.run_analysis('2011-04-27', '2020-06-11')
print(f'Correlation coefficient between news sentiment and {ticker} stock price movements: {correlation_coefficient}')
