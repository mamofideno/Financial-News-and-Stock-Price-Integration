import pandas as pd
import talib
import pynance as pn
import matplotlib.pyplot as plt

class QuantitativeAnalysis:
    def __init__(self, tickers):
        self.tickers = tickers
        self.data = {}

    def fetch_data(self, start_date, end_date):
        for ticker in self.tickers:
            df = pn.data.get(ticker, start=start_date, end=end_date)

            # Adjust column name as per available data
            if 'Adj Close' in df.columns:
                df['Adj_Close'] = df['Adj Close']
            elif 'Close' in df.columns:
                df['Adj_Close'] = df['Close']
            else:
                raise ValueError(f"No adjusted close price found for {ticker}")

            self.data[ticker] = df

    def calculate_technical_indicators(self):
        for ticker in self.tickers:
            df = self.data[ticker]

            df['SMA_20'] = talib.SMA(df['Adj_Close'], timeperiod=20)
            df['SMA_50'] = talib.SMA(df['Adj_Close'], timeperiod=50)

            df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['Adj_Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

            df['RSI'] = talib.RSI(df['Adj_Close'], timeperiod=14)

            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Adj_Close'], fastperiod=12, slowperiod=26, signalperiod=9)

            self.data[ticker] = df

    def analyze(self):
        analysis_summary = {}
        
        for ticker in self.tickers:
            df = self.data[ticker]
            analysis_summary[ticker] = {}

            analysis_summary[ticker]['SMA_Trend'] = df['SMA_20'][-1] > df['SMA_50'][-1]
            analysis_summary[ticker]['RSI'] = df['RSI'][-1]
            analysis_summary[ticker]['MACD_Trend'] = df['MACD'][-1] > df['MACD_Signal'][-1]
            analysis_summary[ticker]['Bollinger_Position'] = 'Upper' if df['Adj_Close'][-1] > df['Upper_BB'][-1] else 'Lower' if df['Adj_Close'][-1] < df['Lower_BB'][-1] else 'Middle'

        return analysis_summary

    def plot_data(self):
        """
        Plots the adjusted close prices for all tickers on the same graph.
        """
        plt.figure(figsize=(14, 7))

        for ticker in self.tickers:
            df = self.data[ticker]
            plt.plot(df.index, df['Adj_Close'], label=ticker)

        plt.title('Adjusted Close Prices of Tickers')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self, start_date, end_date):
        self.fetch_data(start_date, end_date)
        self.calculate_technical_indicators()
        self.plot_data()
        return self.analyze()

# Example usage
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA','META']
qa = QuantitativeAnalysis(tickers)
summary = qa.run('2023-01-01', '2023-08-31')
print(summary)
