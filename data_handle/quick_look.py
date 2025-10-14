import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import date2num
import mplfinance as mpf

import warnings
warnings.filterwarnings('ignore')

class ForexQuickLook():
    def __init__(
        self, 
        data: pd.DataFrame,
        full_look: bool = False,
        periods: int = None,
        prints: bool = True,
        column : str = 'close',
        plots: bool = True,
    ):
        
        """
        Initialize the Forex Data Analyzer
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data
        full_abalysis (bool): Whether to perform full analysis upon initialization  
        periods (int): Number of periods to display
        prints (bool): Whether to print loading information
        column (str): Column to plot
        
        """
        print("="*50)
        print("FOREX QUICK LOOK")
        print("="*50)
        print(" Available Fuctions \n1 display_info \n2 check_missing_values_and_duplicates \n3 plot_candlestick \n4 plot_time_series \n5 get_summary")
        print("="*50)
    
        self.plot = plots    
        self.periods = periods
        self.column = column    
        self.prints = prints    
        self.summary = None
        
        if self.periods is not None:
            if self.periods > len(data):  
                print(f"Warning: periods ({periods}) is greater than the length of the dataframe ({len(data)}). Showing full data instead.")
            else:
                self.data = data.tail(self.periods).copy()
        else:
            self.data = data.copy()
        
        if full_look:
            self.full_analysis()
        else:
            print(self.data.head(3))
            print("="*50)
            print(self.data.tail(3))
            print("="*50)
            print(f"Data shape: {self.data.shape}")
            print("="*50)
            print(f"Columns: {self.data.columns}")
            print("="*50)
            print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
    
    def display_info(self):
        
        """
        Display comprehensive information about the dataset
        
        """
        
        print("DATASET INFORMATION")
        print("="*50)
        
        print("\n1. BASIC INFO:")
        print(self.data.info())
        print("="*50)
        
        print("\n2. FIRST 3 ROWS:")
        print(self.data.head(3))
        print("="*50)
        
        print("\n3. LAST 3 ROWS:")
        print(self.data.tail(3))
        print("="*50)
        
        print("\n4. STATISTICAL DESCRIPTION:")
        print(self.data.describe())
        print("="*50)
        
        print("\n6. DATA TYPES:")
        print(self.data.dtypes)
        print("="*50)
        
        print("\n7. INDEX INFORMATION:")
        print(f"Index name: {self.data.index.name}")
        print(f"Index type: {type(self.data.index)}")
        print("="*50)
    
    def check_missing_values_and_duplicates(self):
        
        """
        Check for missing and NaN values in the dataset
        
        """
        
        print("MISSING VALUES ANALYSIS")
        print("="*50)
        
        nan_values = self.data.isna().sum()
        print("\n1. NaN VALUES PER COLUMN:")
        print(nan_values)
        print("="*50)
        
        duplicate_rows = self.data.duplicated().sum()
        print(f"\n2. DUPLICATE ROWS: {duplicate_rows}")
        print("="*50)
        
        missing_values = self.data.isnull().sum()
        print("\n3. MISSING VALUES PER COLUMN:")
        print(missing_values)
        print("="*50)
        
        total_missing = missing_values.sum()
        print(f"\n4. TOTAL MISSING VALUES: {total_missing}")
        print("="*50)
        
        missing_percentage = (missing_values / len(self.data)) * 100
        print("\n5. MISSING VALUES PERCENTAGE:")
        print(missing_percentage.round(2))
        print("="*50)
        
        infinite_values = np.isinf(self.data.select_dtypes(include=[np.number])).sum()
        print("\n6. INFINITE VALUES:")
        print(infinite_values)
        print("="*50)

    def plot_candlestick(self):
        
        """
        Plot candlestick chart using mplfinance
        
        Parameters:
        periods (int): Number of periods to display
        title (str): Chart title
        
        """
        
        try:
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in self.data.columns for col in required_cols):
                raise ValueError("Error: Missing OHLC columns for candlestick chart")
                
            plt.figure(figsize=(15, 6))
            plot_data = self.data.dropna()
            
            mpf.plot(plot_data, 
                    type='candle',
                    style='charles',
                    title= 'Candlestick Chart',
                    ylabel='Price',
                    volume=False,
                    figratio=(15, 8),
                    figscale=1.2,
                    returnfig=True)
            
            plt.show()  
        except Exception as e:
            print(f"Error creating candlestick chart: {e}")
    
    def plot_time_series(self):
        
        """
        Plot time series of a specific column
        
        Parameters:
        column (str): Column to plot
        periods (int): Number of periods to display
        title (str): Chart title
        
        """
        
        if self.column in self.data.columns:
            plt.figure(figsize=(15, 6))
            
            self.data[self.column].plot()
            plt.title(f'Line Plot of {self.column}')
            plt.ylabel(self.column)
            plt.xlabel('Date')
            plt.grid(True)
            plt.show()
        else:
            print(f"Column '{self.column}' not found in dataset")
    
    def get_summary(
        self,
        prints = True
    ):
        
        """
        Get a comprehensive summary of the dataset
        
        """
        
        summary = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'date_range': f"{self.data.index.min()} to {self.data.index.max()}",
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'non_numeric_columns': list(self.data.select_dtypes(exclude=[np.number]).columns),
            'missing_values': self.data.isnull().sum().sum(),
            'duplicates': self.data.duplicated().sum(),
            'columns' : self.data.columns,
            'index_column': self.data.index.name
        }
        self.summary = summary
        
        if prints == True:
            print("DATASET SUMMARY")
            print("="*50)
            
            for key, value in summary.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
            
            print("="*50)
        
    def full_analysis(self):
       
        """
        Perform full analysis: display info, check missing values, clean data, and get summary
       
        """
       
        self.display_info()
        self.check_missing_values_and_duplicates()
        
        if self.plot:
            self.plot_candlestick()
            self.plot_time_series()
            
        self.get_summary(prints = False) 
        
        return self.data, self.summary