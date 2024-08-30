import pandas as pd
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self, dataframe, headline_column='headline',publisher_column='publisher'):
        """
        Initialize the SentimentAnalyzer with a DataFrame and the column containing headlines.

        :param dataframe: Input DataFrame containing the headlines.
        :param headline_column: The name of the column containing the headlines (default is 'headline').
        """
        self.dataframe = dataframe
        self.headline_column = headline_column
        self.publisher_column = publisher_column

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

    def count_articles_per_publisher(self):
        """
        Count the number of articles per publisher.

        :return: A DataFrame with publishers and their respective article counts.
        """
        # Count the occurrences of each publisher
        publisher_counts = self.dataframe[self.publisher_column].value_counts().reset_index()
        publisher_counts.columns = ['publisher', 'article_count']
        
        return publisher_counts
