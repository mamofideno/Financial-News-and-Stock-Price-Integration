import pandas as pd
import numpy as np
import pynance as pn
from textblob import TextBlob
import matplotlib.pyplot as plt

class MultiTickerNewsStockCorrelation:
    def __init__(self, tickers, news_data_dict):
        """
        Initializes the MultiTickerNewsStockCorrelation class.

        :param tickers: A list of stock ticker symbols (e.g., ['AAPL', 'GOOGL', 'MSFT']).
        :param news_data_dict: A dictionary with ticker symbols as keys and DataFrames with news data as values.
                               Each DataFrame should have two columns: 'Date' and 'Headline'.
        """
        self.tickers = tickers
        self.news_data_dict = news_data_dict
        self.stock_data_dict = {}
        self.weekly_sentiment_scores_dict = {}
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
        Analyzes the sentiment of the news headlines for each ticker using TextBlob and aggregates it weekly.
        """
        for ticker, news_data in self.news_data_dict.items():
            def get_sentiment(text):
                return TextBlob(text).sentiment.polarity

            news_data['Sentiment'] = news_data['headline'].apply(get_sentiment)
            news_data['Date'] = pd.to_datetime(news_data['date'])
            news_data.set_index('date', inplace=True)
            
            # Resample sentiment scores to weekly frequency by averaging
            weekly_sentiment_scores = news_data['Sentiment'].resample('W').mean()
            self.weekly_sentiment_scores_dict[ticker] = weekly_sentiment_scores

    def plot_correlation(self):
        """
        Plots the weekly sentiment scores and weekly stock price changes for all tickers on the same graph.
        """
        plt.figure(figsize=(14, 10))
        
        for ticker in self.tickers:
            plt.subplot(2, 1, 1)
            plt.plot(self.weekly_stock_data_dict[ticker].index, self.weekly_stock_data_dict[ticker], label=f'{ticker} Price Change')
        
        plt.title(f'Weekly Stock Price Change')
        plt.xlabel('Date')
        plt.ylabel('Weekly Price Change')
        plt.grid(True)
        plt.legend()

        for ticker in self.tickers:
            plt.subplot(2, 1, 2)
            plt.plot(self.weekly_sentiment_scores_dict[ticker].index, self.weekly_sentiment_scores_dict[ticker], label=f'{ticker} Sentiment Score')
        
        plt.title(f'Weekly News Sentiment Score')
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

# Sample news data for each ticker
news_data_dict = {
    'AAPL': pd.DataFrame({
        'Date': ['2023-08-01', '2023-08-02', '2023-08-03'],
        'Headline': [
            'Apple releases strong quarterly earnings report',
            'Apple stock surges on new product announcements',
            'Investors worry about Apple’s supply chain issues'
        ]
    }),
    'GOOGL': pd.DataFrame({
        'Date': ['2023-08-01', '2023-08-02', '2023-08-03'],
        'Headline': [
            'Google faces antitrust scrutiny in Europe',
            'Alphabet announces breakthrough in AI technology',
            'Google to expand cloud services in Asia'
        ]
    }),
    'MSFT': pd.DataFrame({
        'Date': ['2023-08-01', '2023-08-02', '2023-08-03'],
        'Headline': [
            'Microsoft partners with OpenAI for new AI solutions',
            'Microsoft reports record revenues in latest quarter',
            'Microsoft to acquire gaming company in billion-dollar deal'
        ]
    }),
    # Add similar data for 4 more tickers...
    'AMZN': pd.DataFrame({
        'Date': ['2023-08-01', '2023-08-02', '2023-08-03'],
        'Headline': [
            'Amazon expands into new markets with innovative strategies',
            'Amazon faces challenges with supply chain disruptions',
            'Amazon to introduce new product line next quarter'
        ]
    }),
    'TSLA': pd.DataFrame({
        'Date': ['2023-08-01', '2023-08-02', '2023-08-03'],
        'Headline': [
            'Tesla unveils new electric vehicle model',
            'Tesla stock drops amid regulatory concerns',
            'Elon Musk announces new Tesla factory location'
        ]
    }),
    'NFLX': pd.DataFrame({
        'Date': ['2023-08-01', '2023-08-02', '2023-08-03'],
        'Headline': [
            'Netflix releases highly anticipated new series',
            'Netflix faces increased competition from streaming rivals',
            'Netflix announces expansion into new markets'
        ]
    }),
    'FB': pd.DataFrame({
        'Date': ['2023-08-01', '2023-08-02', '2023-08-03'],
        'Headline': [
            'Facebook rebrands to focus on the metaverse',
            'Facebook faces backlash over privacy issues',
            'Meta (Facebook) announces new virtual reality products'
        ]
    })
}

# Initialize the correlation analysis class
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NFLX', 'FB']
correlation_analysis = MultiTickerNewsStockCorrelation(tickers, news_data_dict)

# Run the analysis
correlation_analysis.run_analysis('2023-07-01', '2023-08-31')
