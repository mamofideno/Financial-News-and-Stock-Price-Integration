import yfinance as yf
import talib
import matplotlib.pyplot as plt

class StockAnalyzer:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}

    def download_data(self):
        self.data = {ticker: yf.download(ticker, start=self.start_date, end=self.end_date) for ticker in self.tickers}

    def calculate_indicators(self):
        for ticker in self.tickers:
            df = self.data[ticker]
            df['SMA'] = talib.SMA(df['Close'], timeperiod=30)
            df['EMA'] = talib.EMA(df['Close'], timeperiod=30)
            df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20)
            df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
            df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)

    def plot_indicators(self):
        for ticker in self.tickers:
            df = self.data[ticker]
            
            plt.figure(figsize=(14, 8))
            plt.plot(df.index, df['Close'], label=f'{ticker} Close')
            plt.plot(df.index, df['SMA'], label=f'{ticker} SMA')
            plt.title(f'{ticker} - Simple Moving Average (SMA)')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.show()

            plt.figure(figsize=(14, 8))
            plt.plot(df.index, df['Close'], label=f'{ticker} Close')
            plt.plot(df.index, df['EMA'], label=f'{ticker} EMA')
            plt.title(f'{ticker} - Exponential Moving Average (EMA)')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.show()

            plt.figure(figsize=(14, 8))
            plt.plot(df.index, df['RSI'], label=f'{ticker} RSI')
            plt.title(f'{ticker} - Relative Strength Index (RSI)')
            plt.xlabel('Date')
            plt.ylabel('RSI')
            plt.legend()
            plt.show()

            plt.figure(figsize=(14, 8))
            plt.plot(df.index, df['MACD'], label=f'{ticker} MACD')
            plt.plot(df.index, df['MACD_signal'], label=f'{ticker} MACD Signal')
            plt.bar(df.index, df['MACD_hist'], label=f'{ticker} MACD Hist')
            plt.title(f'{ticker} - Moving Average Convergence Divergence (MACD)')
            plt.xlabel('Date')
            plt.ylabel('MACD')
            plt.legend()
            plt.show()

            plt.figure(figsize=(14, 8))
            plt.plot(df.index, df['Close'], label=f'{ticker} Close')
            plt.plot(df.index, df['BB_upper'], label=f'{ticker} BB Upper')
            plt.plot(df.index, df['BB_middle'], label=f'{ticker} BB Middle')
            plt.plot(df.index, df['BB_lower'], label=f'{ticker} BB Lower')
            plt.title(f'{ticker} - Bollinger Bands (BB)')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.show()

            plt.figure(figsize=(14, 8))
            plt.plot(df.index, df['ATR'], label=f'{ticker} ATR')
            plt.title(f'{ticker} - Average True Range (ATR)')
            plt.xlabel('Date')
            plt.ylabel('ATR')
            plt.legend()
            plt.show()

            plt.figure(figsize=(14, 8))
            plt.plot(df.index, df['ADX'], label=f'{ticker} ADX')
            plt.title(f'{ticker} - Average Directional Index (ADX)')
            plt.xlabel('Date')
            plt.ylabel('ADX')
            plt.legend()
            plt.show()

    def analyze(self):
        self.download_data()
        self.calculate_indicators()
        self.plot_indicators()

# Example usage: