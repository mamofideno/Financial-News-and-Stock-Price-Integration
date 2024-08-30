import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class HeadlineStatistics:
    def __init__(self, df, headline_column):
        """
        Initialize the HeadlineStatistics class.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        headline_column (str): The name of the column containing the headlines.
        """
        self.df = df
        self.headline_column = headline_column
        self.df['headline_length'] = self.df[self.headline_column].apply(len)
        
    def calculate_statistics(self):
        """
        Calculate basic statistics for headline lengths.
        
        Returns:
        pd.Series: Summary statistics including count, mean, std, min, 25%, 50%, 75%, and max.
        """
        return self.df['headline_length'].describe()
    
    def plot_length_distribution(self, bins=20):
        """
        Plot the distribution of headline lengths.
        
        Parameters:
        bins (int): Number of bins for the histogram. Default is 20.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['headline_length'], bins=bins, kde=True)
        plt.title('Distribution of Headline Lengths')
        plt.xlabel('Headline Length (characters)')
        plt.ylabel('Frequency')
        plt.show()
    
    def get_outliers(self, threshold=1.5):
        """
        Identify outliers in headline lengths using the IQR method.
        
        Parameters:
        threshold (float): The threshold multiplier for determining outliers. Default is 1.5.
        
        Returns:
        pd.DataFrame: A DataFrame containing the outlier headlines and their lengths.
        """
        Q1 = self.df['headline_length'].quantile(0.25)
        Q3 = self.df['headline_length'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = self.df[(self.df['headline_length'] < lower_bound) | (self.df['headline_length'] > upper_bound)]
        return outliers[[self.headline_column, 'headline_length']]
    
    def plot_boxplot(self):
        """
        Plot a boxplot of headline lengths to visualize the distribution and outliers.
        """
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=self.df['headline_length'])
        plt.title('Boxplot of Headline Lengths')
        plt.xlabel('Headline Length (characters)')
        plt.show()
