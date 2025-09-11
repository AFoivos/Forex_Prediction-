import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import date2num
import mplfinance as mpf
import warnings
warnings.filterwarnings('ignore')

class FirstLook:
    def __init__(self, file_path):
        """
        Initialize the Forex Data Analyzer
        
        Parameters:
        file_path (str): Path to the CSV file
        """
        self.file_path = file_path
        self.data = None
        self.load_data()
        self.get_data()
    
    def load_data(self):
        """
        Load CSV data and perform initial processing !!! MUST BE FROM METATRADER 5 !!!
        """
        try:
            # Load the CSV file
            new_column_names = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
            self.data = pd.read_csv(self.file_path,
                                        header=None,   
                                        sep='\t',
                                        names=new_column_names, 
                                        skiprows=1  
                                        )
            self.data['datetime'] = pd.to_datetime(self.data['date'] + ' ' + self.data['time'])
    
            self.data.drop(columns=['tickvol', 
                            'vol', 
                            'spread'],
                            inplace=True)
    
            self.data = self.data[self.data['datetime'].dt.year > 2020]
    
            self.data.set_index('datetime', inplace=True)
            
            # Display initial information
            print('Data loaded successfully!')
            print(f'Shape: {self.data.shape}')
            print("\n" + "="*50)
            
        except Exception as e:
            print(f"Error loading file: {e}")
    
    def display_info(self):
        """
        Display comprehensive information about the dataset
        """
        print("DATASET INFORMATION")
        print("="*50)
        
        # Basic info
        print("\n1. BASIC INFO:")
        print(self.data.info())
        print("="*50)
        
        # Head of the data
        print("\n2. FIRST 3 ROWS:")
        print(self.data.head(3))
        print("="*50)
        
        # Tail of the data
        print("\n3. LAST 3 ROWS:")
        print(self.data.tail(3))
        print("="*50)
        
        # Statistical description
        print("\n4. STATISTICAL DESCRIPTION:")
        print(self.data.describe())
        print("="*50)
        
        # Data types
        print("\n6. DATA TYPES:")
        print(self.data.dtypes)
        print("="*50)
        
        # Index information
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
        
        # Check for NaN values
        nan_values = self.data.isna().sum()
        print("\n1. NaN VALUES PER COLUMN:")
        print(nan_values)
        print("="*50)
        
        # Check for Duplicate values
        duplicate_rows = self.data.duplicated().sum()
        print(f"\n2. DUPLICATE ROWS: {duplicate_rows}")
        print("="*50)
        
        # Total missing values per column
        missing_values = self.data.isnull().sum()
        print("\n3. MISSING VALUES PER COLUMN:")
        print(missing_values)
        print("="*50)
        
        # Total missing values
        total_missing = missing_values.sum()
        print(f"\n4. TOTAL MISSING VALUES: {total_missing}")
        print("="*50)
        
        # Percentage of missing values
        missing_percentage = (missing_values / len(self.data)) * 100
        print("\n5. MISSING VALUES PERCENTAGE:")
        print(missing_percentage.round(2))
        print("="*50)
        
        # Check for infinite values
        infinite_values = np.isinf(self.data.select_dtypes(include=[np.number])).sum()
        print("\n6. INFINITE VALUES:")
        print(infinite_values)
        print("="*50)
    
    def clean_data(self, show_print=True):
        """
        Clean the data by handling missing values and duplicates
        """
        if show_print == True   :  
            print("DATA CLEANING")
            print("="*50)
        
        # Remove duplicates
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates()
        duplicates_removed = initial_rows - len(self.data)
        
        if show_print == True:  
            print(f"Duplicates removed: {duplicates_removed}")
        
        # Handle missing values
        missing_before = self.data.isnull().sum().sum()
        
        # For numeric columns, fill with forward fill then backward fill
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        missing_after = self.data.isnull().sum().sum()
        
        if show_print == True:  
            print(f"Missing values handled: {missing_before - missing_after}")
            
            print("Data cleaning completed!")
    
    def plot_candlestick(self, periods=100, title="Forex Candlestick Chart"):
        """
        Plot candlestick chart using mplfinance
        
        Parameters:
        periods (int): Number of periods to display
        title (str): Chart title
        """
        try:
            # Ensure we have OHLC data
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in self.data.columns for col in required_cols):
                print("Error: Missing OHLC columns for candlestick chart")
                return
            
            # Get the last 'periods' data points
            plot_data = self.data.iloc[-periods:].copy()
            
            # Create the candlestick chart
            plt.figure(figsize=(15, 6))
            
            mpf.plot(plot_data, 
                    type='candle',
                    style='charles',
                    title=title,
                    ylabel='Price',
                    volume=False,
                    figratio=(15, 8),
                    figscale=1.2,
                    returnfig=True)
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating candlestick chart: {e}")
    
    def plot_time_series(self, column='close', periods=200, title="Price Time Series"):
        """
        Plot time series of a specific column
        
        Parameters:
        column (str): Column to plot
        periods (int): Number of periods to display
        title (str): Chart title
        """
        if column in self.data.columns:
            plt.figure(figsize=(15, 6))
            self.data[column].iloc[-periods:].plot()
            plt.title(f'{title} - Last {periods} periods')
            plt.ylabel(column)
            plt.xlabel('Date')
            plt.grid(True)
            plt.show()
        else:
            print(f"Column '{column}' not found in dataset")
    
    def get_summary(self):
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
        }
        
        print("DATASET SUMMARY")
        print("="*50)
        
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("="*50)
        
    def get_data(self):
        """
        Returns the processed DataFrame.
        """
        data = self.data
        cleaned_data = self.clean_data(show_print=False)
        summary = self.get_summary()
        
        return cleaned_data, summary
