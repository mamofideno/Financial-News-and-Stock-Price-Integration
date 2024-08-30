import pandas as pd
import matplotlib.pyplot as plt

class Histogram:
    def __init__(self, dataframe, column_name,x_label,y_label='Frequency',title='' ):
        self.dataframe = dataframe
        self.column_name = column_name
        self.x_label=x_label
        self.y_label=y_label
        self.title=title

    def plot_histogram(self,size=(200,50)):
        # Check if the column exists in the DataFrame
        if self.column_name not in self.dataframe.columns:
            raise ValueError(f"Column '{self.column_name}' does not exist in the DataFrame.")
        
        # Plot histogram
        self.dataframe[self.column_name].value_counts().plot(kind='bar')
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.plot(size)
        plt.show()