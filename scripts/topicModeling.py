import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

class TopicModeling:
    def __init__(self, dataframe, headline_column='headline'):
        """
        Initialize the NLPAnalyzer with a DataFrame and the column containing headlines.

        :param dataframe: Input DataFrame containing the headlines.
        :param headline_column: The name of the column containing the headlines (default is 'headline').
        """
        self.dataframe = dataframe
        self.headline_column = headline_column
        self.stop_words = set(stopwords.words('english') + list(string.punctuation))

    def preprocess_text(self, text):
        """
        Preprocess the text by converting to lowercase, removing punctuation and stopwords.

        :param text: The text to preprocess.
        :return: The cleaned and tokenized text.
        """
        text = text.lower()
        tokens = [word for word in text.split() if word not in self.stop_words]
        return ' '.join(tokens)

    def extract_keywords(self, ngram_range=(1, 2), top_n=10):
        """
        Extract the most common keywords or phrases (n-grams) from the headlines.

        :param ngram_range: The range of n-grams to consider (default is unigrams and bigrams).
        :param top_n: The number of top keywords/phrases to return (default is 10).
        :return: A DataFrame of the most common n-grams.
        """
        self.dataframe['cleaned_headline'] = self.dataframe[self.headline_column].apply(self.preprocess_text)
        
        # Vectorize the text to get n-grams
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        X = vectorizer.fit_transform(self.dataframe['cleaned_headline'])

        # Get the sum of each n-gram across all documents
        ngram_counts = X.sum(axis=0).tolist()[0]
        ngram_features = vectorizer.get_feature_names_out()

        # Create a DataFrame of n-grams and their counts
        ngram_df = pd.DataFrame({'ngram': ngram_features, 'count': ngram_counts})
        ngram_df = ngram_df.sort_values(by='count', ascending=False).head(top_n)

        return ngram_df

    def perform_topic_modeling(self, num_topics=5, num_words=10):
        """
        Perform topic modeling using LDA to identify significant topics in the headlines.

        :param num_topics: The number of topics to identify (default is 5).
        :param num_words: The number of words to display per topic (default is 10).
        :return: A list of topics, each represented by a list of words.
        """
        self.dataframe['cleaned_headline'] = self.dataframe[self.headline_column].apply(self.preprocess_text)
        
        # Vectorize the text
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(self.dataframe['cleaned_headline'])

        # Apply LDA
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(X)

        # Extract the topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-num_words - 1:-1]]
            topics.append(top_words)

        return topics
