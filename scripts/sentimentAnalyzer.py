import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentAnalyzer:
    def __init__(self, dataframe, headline_column='headline', publisher_column='publisher',date_column='date'):
        """
        Initialize the SentimentAnalyzer with a DataFrame and the column containing headlines.

        :param dataframe: Input DataFrame containing the headlines.
        :param headline_column: The name of the column containing the headlines (default is 'headline').
        """
        self.dataframe = dataframe
        self.headline_column = headline_column
        self.publisher_column = publisher_column
        self.date_column=date_column
        self.dataframe[self.date_column] = pd.to_datetime(self.dataframe[self.date_column],errors='coerce')

    def calculate_sentiment(self):
        """
        Calculate the sentiment of each headline in the DataFrame and return a new DataFrame with a sentiment column.

        :return: DataFrame with an added 'sentiment' column containing the polarity of each headline.
        """
        # Define a function to apply TextBlob's sentiment analysis
        def get_sentiment(text):
            return TextBlob(text).sentiment.polarity
        
        def categorize_sentiment(score):
            if score > 0.1:
                return "Positive"
            elif score < -0.1:
                return "Negative"
            else:
                return "Neutral"
        
        # Apply the sentiment function to the headline column
        self.dataframe['sentiment_score'] = self.dataframe[self.headline_column].apply(get_sentiment)
        
        # Categorize the sentiment score
        self.dataframe['sentiment'] = self.dataframe['sentiment_score'].apply(categorize_sentiment)
        
        
        return self.dataframe

    # def count_articles_per_publisher(self):
    #     """
    #     Count the number of articles per publisher.

    #     :return: A DataFrame with publishers and their respective article counts.
    #     """
    #     # Count the occurrences of each publisher
    #     publisher_counts = self.dataframe[self.publisher_column].value_counts().reset_index()
    #     publisher_counts.columns = ['publisher', 'article_count']
        
    #     return publisher_counts
    def count_articles_per_publisher(self):
        """
        Count the number of articles per publisher.
        
        Returns:
        pd.DataFrame: A DataFrame with publishers and their corresponding article counts.
        """
        publisher_counts = self.dataframe['publisher'].value_counts().reset_index()
        publisher_counts.columns = ['publisher', 'article_count']
        return publisher_counts
    
    def plot_publisher_activity(self, top_n=None):
        """
        Plot the number of articles per publisher.
        
        Parameters:
        top_n (int): The number of top publishers to plot. If None, plot all.
        """
        publisher_counts = self.count_articles_per_publisher()
        
        if top_n:
            publisher_counts = publisher_counts.head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=publisher_counts, x='article_count', y=self.publisher_column, palette='viridis')
        plt.title('Number of Articles per Publisher')
        plt.xlabel('Article Count')
        plt.ylabel('Publisher')
        plt.show()
    
    def get_most_active_publishers(self, top_n=5):
        """
        Get the top N most active publishers by article count.
        
        Parameters:
        top_n (int): The number of top publishers to return.
        
        Returns:
        pd.DataFrame: A DataFrame containing the top N most active publishers.
        """
        publisher_counts = self.count_articles_per_publisher()
        return publisher_counts.head(top_n)
    def aggregate_by(self, freq='D'):
        """
        Aggregate the number of publications by a specified frequency.
        
        Parameters:
        freq (str): Frequency string for resampling (e.g., 'D' for daily, 'W' for weekly, 'M' for monthly).
        
        Returns:
        pd.DataFrame: A DataFrame with the aggregated publication counts.
        """
        aggregated_data = self.dataframe[self.date_column].value_counts().resample(freq).sum().fillna(0)
        return aggregated_data
    
    def plot_publication_trend(self, freq='D'):
        """
        Plot the publication trend over time.
        
        Parameters:
        freq (str): Frequency string for resampling (e.g., 'D' for daily, 'W' for weekly, 'M' for monthly).
        """
        aggregated_data = self.aggregate_by(freq)
        
        plt.figure(figsize=(12, 6))
        aggregated_data.plot()
        plt.title(f'Publication Trend Over Time ({freq} Frequency)')
        plt.xlabel('Date')
        plt.ylabel('Number of Publications')
        plt.grid(True)
        plt.show()
    
    def detect_peak_days(self, threshold=None, top_n=None):
        """
        Detect peak days with unusually high numbers of publications.
        
        Parameters:
        threshold (int): Minimum number of publications to consider a peak day. If None, use top_n instead.
        top_n (int): The number of top days with the highest publication counts to return. Ignored if threshold is set.
        
        Returns:
        pd.DataFrame: A DataFrame with the peak days and their publication counts.
        """
        daily_counts = self.aggregate_by('D')
        
        if threshold:
            peak_days = daily_counts[daily_counts > threshold].sort_values(ascending=False)
        elif top_n:
            peak_days = daily_counts.sort_values(ascending=False).head(top_n)
        else:
            raise ValueError("Either 'threshold' or 'top_n' must be provided.")
        
        return peak_days.reset_index().rename(columns={self.date_column: 'date', 0: 'publication_count'})
    
    def plot_weekday_distribution(self):
        """
        Plot the distribution of publications by weekday.
        """
        self.dataframe['weekday'] = self.dataframe[self.date_column].dt.day_name()
        weekday_counts = self.dataframe['weekday'].value_counts().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=weekday_counts.index, y=weekday_counts.values, palette='coolwarm')
        plt.title('Distribution of Publications by Weekday')
        plt.xlabel('Weekday')
        plt.ylabel('Number of Publications')
        plt.show()

    def plot_monthly_distribution(self):
        """
        Plot the distribution of publications by month.
        """
        self.dataframe['month'] = self.dataframe[self.date_column].dt.month_name()
        monthly_counts = self.dataframe['month'].value_counts().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ])
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=monthly_counts.index, y=monthly_counts.values, palette='magma')
        plt.title('Distribution of Publications by Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Publications')
        plt.show()
    
