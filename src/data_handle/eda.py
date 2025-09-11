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
    
    def basic_analysis(self, plot=True):
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
    def price_trend_analysis(self, price_column='close', plot=True):
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
        
        if plot == True:
            try:
                plt.figure(figsize=(12, 6))
                plt.plot(prices.tail(48).index, prices.tail(48).values, label='Price', linewidth=1.5, alpha=0.8)
                plt.plot(short_ma.tail(48).index, short_ma.tail(48).values, label='20-period MA', linewidth=2, color='red')
                plt.plot(long_ma.tail(48).index, long_ma.tail(48).values, label='50-period MA', linewidth=2, color='blue')
                plt.title(f'Price with Moving Averages - {current_trend}')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
            except Exception as e:
                print(f"Error in plotting price trend analysis: {e}")
        
        # Current trend
        current_trend = "Uptrend" if short_ma.iloc[-1] > long_ma.iloc[-1] else "Downtrend"
        print(f"\nCurrent trend (20 vs 50 MA): {current_trend}")
        
        if len(prices) > 50:
            # Calculate crossover signals
            crossover_up = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
            crossover_down = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
            
            print(f"Bullish crossovers (20MA > 50MA): {crossover_up.sum()}")
            print(f"Bearish crossovers (20MA < 50MA): {crossover_down.sum()}")
            
            # Recent crossover
            last_crossover = "Bullish" if crossover_up.iloc[-1] else ("Bearish" if crossover_down.iloc[-1] else "None")
            print(f"Most recent crossover: {last_crossover}")

    def stationarity_tests(self, column='close', plot=True, lags=40):
        """
        Perform stationarity tests (ADF and KPSS)
        """        
        if column not in self.data.columns:
            print(f"Column '{column}' not found!")
            return
        print(f"\nSTATIONARITY TESTS ({column.upper()})")
        print("=" * 60)
        
        series = self.data[column].dropna()
        
        # Augmented Dickey-Fuller test
        print("Augmented Dickey-Fuller Test:")
        adf_result = adfuller(series)
        print(f'ADF Statistic: {adf_result[0]:.4f}')
        print(f'p-value: {adf_result[1]:.4f}')
        
        # KPSS test
        print("\nKPSS Test:")
        kpss_result = kpss(series, regression='c')
        print(f'KPSS Statistic: {kpss_result[0]:.4f}')
        print(f'p-value: {kpss_result[1]:.4f}')
        
        # Interpretation
        print("\nINTERPRETATION:")
        if adf_result[1] <= 0.05:
            print("✓ ADF: Series is stationary (reject null hypothesis)")
            adf_stationary = True
        else:
            print("✗ ADF: Series is non-stationary (cannot reject null hypothesis)")
            adf_stationary = False
            
        if kpss_result[1] >= 0.05:
            print("✓ KPSS: Series is stationary (cannot reject null hypothesis)")
            kpss_stationary = True
        else:
            print("✗ KPSS: Series is non-stationary (reject null hypothesis)")
            kpss_stationary = False
            
        if plot == True:
            try:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Stationarity Analysis - {column.upper()}\n'
                                f'ADF: {"Stationary" if adf_stationary else "Non-Stationary"} | '
                                f'KPSS: {"Stationary" if kpss_stationary else "Non-Stationary"}', 
                                fontsize=16, fontweight='bold')
                    
                # 1. Original Series
                ax1.plot(series.index, series.values, linewidth=2)
                ax1.set_title('Original Time Series')
                ax1.set_ylabel('Price')                
                
                ax1.grid(True, alpha=0.3)
                
                # 2. Rolling Mean and Std
                rolling_mean = series.rolling(window=30).mean()
                rolling_std = series.rolling(window=30).std()
                    
                ax2.plot(series.index, series.values, label='Original', alpha=0.7)
                ax2.plot(rolling_mean.index, rolling_mean.values, label='Rolling Mean (30)', linewidth=2, color='red')
                ax2.plot(rolling_std.index, rolling_std.values, label='Rolling Std (30)', linewidth=2, color='green')
                ax2.set_title('Rolling Statistics')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                    
                # 3. First Difference
                first_diff = series.diff().dropna()
                ax3.plot(first_diff.index, first_diff.values, linewidth=2, color='purple')
                ax3.set_title('First Difference')
                ax3.set_ylabel('Difference')
                ax3.grid(True, alpha=0.3)
                    
                # 4. Histogram of First Difference
                ax4.hist(first_diff, bins=50, alpha=0.7, density=True, color='orange')
                ax4.set_title('Distribution of First Differences')
                ax4.set_xlabel('Difference')
                ax4.set_ylabel('Density')
                ax4.grid(True, alpha=0.3)
                    
                # Add some statistics to the histogram
                ax4.axvline(first_diff.mean(), color='red', linestyle='--', label=f'Mean: {first_diff.mean():.4f}')
                ax4.axvline(first_diff.mean() + first_diff.std(), color='green', linestyle='--', 
                        label=f'±1 STD: {first_diff.std():.4f}')
                ax4.axvline(first_diff.mean() - first_diff.std(), color='green', linestyle='--')
                ax4.legend()
                
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error in plotting stationarity analysis: {e}")

        
    def seasonal_decomposition(self, column='close', period=30, plot=True):
        """
        Basic seasonal decomposition analysis
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        if column not in self.data.columns:
            print(f"Column '{column}' not found!")
            return
            
        print(f"\nSEASONAL DECOMPOSITION ({column.upper()})")
        print("=" * 60)
        
        series = self.data[column].dropna()
        if plot == True:
            try:
                result = seasonal_decompose(series, period=period, model='additive')
                
                plt.figure(figsize=(15, 12))
                
                plt.subplot(4, 1, 1)
                plt.plot(result.observed)
                plt.title('Observed')
                
                plt.subplot(4, 1, 2)
                plt.plot(result.trend)
                plt.title('Trend')
                
                plt.subplot(4, 1, 3)
                plt.plot(result.seasonal)
                plt.title('Seasonal')
                
                plt.subplot(4, 1, 4)
                plt.plot(result.resid)
                plt.title('Residual')
                
                plt.tight_layout()
                plt.show()       
            except Exception as e:
                print(f"Error in plotting seasonal decomposition: {e}")
        
    def autocorrelation_analysis(self, column='close', lags=40, plot=True):
        """
        Analyze autocorrelation and partial autocorrelation
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found!")
            return
            
        print(f"\nAUTOCORRELATION ANALYSIS ({column.upper()})")
        print("=" * 60)
        
        series = self.data[column].dropna()
        
        if plot == True:
            try:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
                
                # ACF plot
                plot_acf(series, lags=lags, ax=ax1)
                ax1.set_title(f'Autocorrelation Function (ACF) - {column.upper()}')
                
                # PACF plot
                plot_pacf(series, lags=lags, ax=ax2)
                ax2.set_title(f'Partial Autocorrelation Function (PACF) - {column.upper()}')
                
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error in plotting ACF/PACF: {e}")
        
    def distribution_analysis(self, column='close', plot=True):
        """
        Analyze distribution of prices or returns
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found!")
            return
            
        print(f"\nDISTRIBUTION ANALYSIS ({column.upper()})")
        print("=" * 60)
        
        series = self.data[column].dropna()
        returns = series.pct_change().dropna()
        
        if plot == True:
            try:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                
                # Price distribution
                ax1.hist(series, bins=50, alpha=0.7, density=True)
                ax1.set_title(f'{column.upper()} Distribution')
                ax1.set_xlabel('Price')
                ax1.set_ylabel('Density')
                
                # Returns distribution
                ax2.hist(returns, bins=50, alpha=0.7, density=True)
                ax2.set_title('Returns Distribution')
                ax2.set_xlabel('Returns')
                ax2.set_ylabel('Density')
                
                # QQ plot for prices
                stats.probplot(series, dist="norm", plot=ax3)
                ax3.set_title('QQ Plot - Prices')
                
                # QQ plot for returns
                stats.probplot(returns, dist="norm", plot=ax4)
                ax4.set_title('QQ Plot - Returns')
                
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error in plotting distributions: {e}")
        
        # Normality tests
        print("Normality Tests for Returns:")
        stat, p = stats.normaltest(returns)
        print(f'D\'Agostino\'s K^2 Test: statistic={stat:.4f}, p-value={p:.4f}')
        
        stat, p = stats.shapiro(returns)
        print(f'Shapiro-Wilk Test: statistic={stat:.4f}, p-value={p:.4f}')
        
    def volatility_analysis(self, price_column='close', window=20, plot=True):
        """
        Analyze volatility patterns
        """
        if price_column not in self.data.columns:
            print(f"Column '{price_column}' not found!")
            return
            
        print(f"\nVOLATILITY ANALYSIS ({price_column.upper()})")
        print("=" * 60)
        
        returns = self.data[price_column].pct_change().dropna()
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        if plot == True:
            try:
                plt.figure(figsize=(15, 8))
                
                plt.subplot(2, 1, 1)
                plt.plot(returns.index, returns * 100, alpha=0.7)
                plt.title('Daily Returns (%)')
                plt.ylabel('Returns %')
                plt.grid(True)
                
                plt.subplot(2, 1, 2)
                plt.plot(rolling_vol.index, rolling_vol * 100)
                plt.title(f'Rolling {window}-Day Annualized Volatility (%)')
                plt.ylabel('Volatility %')
                plt.grid(True)
                
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error in plotting volatility: {e}")
        
        print(f"Average volatility: {rolling_vol.mean() * 100:.2f}%")
        print(f"Maximum volatility: {rolling_vol.max() * 100:.2f}%")
        print(f"Minimum volatility: {rolling_vol.min() * 100:.2f}%")
        
        
    def correlation_analysis(self, plot=True):
        """
        Analyze correlations between different columns
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("Not enough numeric columns for correlation analysis")
            return
            
        print("\nCORRELATION ANALYSIS")
        print("=" * 60)
        
        # Correlation matrix
        corr_matrix = self.data[numeric_cols].corr()
        
        if plot == True:
            try:
                plt.figure(figsize=(12, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                        square=True, fmt='.3f')
                plt.title('Correlation Matrix')
                plt.show()
            except Exception as e:
                print(f"Error in plotting correlation matrix: {e}")
            
        # Print strongest correlations
        print("Strongest correlations:")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # correlation threshold
                    print(f"{col1} - {col2}: {corr_value:.3f}")
        
        
    def comprehensive_eda(self,plot=True):
        """
        Run comprehensive EDA analysis
        """
        print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        self.basic_analysis(plot=plot)
        self.price_trend_analysis('close', plot=plot)
        self.stationarity_tests('close', plot=plot)
        self.distribution_analysis('close', plot=plot)
        self.volatility_analysis('close', plot=plot)
        self.correlation_analysis(plot=plot)
        self.autocorrelation_analysis('close', plot=plot)
        
        print("\n EDA analysis completed!")