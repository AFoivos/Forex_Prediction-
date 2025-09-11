import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))

if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.data_handle.load_clean_first_look import FirstLook

import warnings
warnings.filterwarnings('ignore')

class ForexEDA(FirstLook):
    def __init__(self, file_path, datetime_column='datetime'):
        super().__init__(file_path)  
        self.get_data()
        self.datetime_column = datetime_column
    
    def basic_analysis(self):
        """
        Perform basic data analysis
        """
        print("BASIC DATA ANALYSIS")
        print("=" * 60)
        
        # Basic information
        print(f"Dataset shape: {self.data.shape}")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Total periods: {len(self.data)}")
        
        # Check for OHLC columns
        ohlc_cols = ['open', 'high', 'low', 'close']
        available_ohlc = [col for col in ohlc_cols if col in self.data.columns]
        print(f"Available OHLC columns: {available_ohlc}")
        
        # Data types
        print("\nData types:")
        print(self.data.dtypes)
        
        # Missing values
        print("\nMissing values:")
        print(self.data.isnull().sum())
    
    # Trend Analysis
    def price_trend_analysis(self, price_column='close'):
        """
        Analyze price trends and patterns
        """
        if price_column not in self.data.columns:
            print(f"Column '{price_column}' not found!")
            return
            
        print(f"\nPRICE TREND ANALYSIS ({price_column.upper()})")
        print("=" * 60)
        
        prices = self.data[price_column]
        
        # Basic statistics
        print(f"Mean price: {prices.mean():.4f}")
        print(f"Median price: {prices.median():.4f}")
        print(f"Standard deviation: {prices.std():.4f}")
        print(f"Minimum price: {prices.min():.4f}")
        print(f"Maximum price: {prices.max():.4f}")
        
        # Price changes
        returns = prices.pct_change().dropna()
        print(f"\nAverage daily return: {returns.mean() * 100:.4f}%")
        print(f"Return volatility: {returns.std() * 100:.4f}%")
        print(f"Maximum daily gain: {returns.max() * 100:.4f}%")
        print(f"Maximum daily loss: {returns.min() * 100:.4f}%")
        
        # Trend indicators
        short_ma = prices.rolling(window=20).mean()
        long_ma = prices.rolling(window=50).mean()
        
        # Current trend
        current_trend = "Uptrend" if short_ma.iloc[-1] > long_ma.iloc[-1] else "Downtrend"
        print(f"\nCurrent trend (20 vs 50 MA): {current_trend}")

    def stationarity_tests(self, column='close'):
        """Perform stationarity tests"""
        pass
        
    def seasonal_decomposition(self, column='close', period=30):
        """Seasonal decomposition"""
        pass
        
    def autocorrelation_analysis(self, column='close', lags=40):
        """Autocorrelation analysis"""
        pass
        
    def distribution_analysis(self, column='close'):
        """Distribution analysis"""
        pass
        
    def volatility_analysis(self, price_column='close', window=20):
        """Volatility analysis"""
        pass
        
    def correlation_analysis(self):
        """Correlation analysis"""
        pass
        
    def comprehensive_eda(self):
        """Comprehensive EDA"""
        pass