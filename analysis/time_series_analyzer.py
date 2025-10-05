import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexTimeSeriesAnalyzer:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
        period: int = None
    ):
        
        """
        Class for Time Series Analysis
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
        print("="*50)
        print("TIME SERIES ANALYSIS")
        print("="*50)
        print(" Available Functions \n1 basic_descriptive_stats \n2 stationarity_tests \n3 distribution_analysis \n4 autocorrelation_analysis \n5 volatility_analysis \n6 drawdown_analysis \n7 seasonal_decomposition \n8 run_complete_analysis")
        print("="*50)
        
        self.data = data.copy() if period == None else data.tail(period).copy()
        self.close_col = close_col
        
        self.parameters = {}
        
        self.analysis_results = pd.DataFrame(
            {self.close_col: self.data[self.close_col]},
            index=self.data.index
        )
        
        self._validate_columns()
        
    def _validate_columns(
        self, 
        columns: list[str] = None,
    ):  
        """
        Validate that required columns exist
        
        Parameters:
        columns (list[str]): List of column names to validate
            
        """
        
        required_cols = [
            self.close_col,
        ]
        
        if columns is not None:
            required_cols.extend(columns)
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    def _is_nested_list(
        self, 
        lst
    ):
        
        """
        Check if a list is nested or not
        
        Parameters:
        lst (list): List to check
        
        """
        
        if not all(isinstance(item, list) for item in lst):
            return [lst]
        else:
            return lst
    
    def basic_descriptive_stats(
        self,
        prints = True
    ):
        
        """
        Basic Descriptive Statistics
        
        """
    
        self.analysis_results['returns'] = self.data[self.close_col].pct_change()
        
        stats_data = {
            'Price': self.data[self.close_col],
            'Returns': self.analysis_results['returns'].dropna(),
            #we add some indicators too
        }
        
        descriptive_stats = {}
        
        for name, series in stats_data.items():
            if prints:
                print(f"\n{name}:")
                print(f"  Count: {series.count():,.2f}")
                print(f"  Mean: {series.mean():.2f}")
                print(f"  Std: {series.std():.2f}")
                print(f"  Min: {series.min():.2f}")
                print(f"  25%: {series.quantile(0.25):.2f}")
                print(f"  50%: {series.quantile(0.50):.2f}")
                print(f"  75%: {series.quantile(0.75):.2f}")
                print(f"  Max: {series.max():.2f}")
                print(f"  Skewness: {series.skew():.2f}")
                print(f"  Kurtosis: {series.kurtosis():.2f}")
            
            descriptive_stats[name] = {
                'count': series.count(),
                'mean': round(series.mean(), 4),
                'std': round(series.std(), 4),
                'min': round(series.min(), 4),
                '25%': round(series.quantile(0.25), 4),
                '50%': round(series.quantile(0.50), 4),
                '75%': round(series.quantile(0.75), 4),
                'max': round(series.max(), 4),
                'skewness': round(series.skew(), 4),
                'kurtosis': round(series.kurtosis(), 4)
            }
        
        self.parameters['descriptive_stats'] = descriptive_stats
        
        return self.analysis_results
    
    def stationarity_tests(
        self,
    ):
        
        """
        Stationarity Tests (ADF, KPSS)
        
        """
        
        series_to_test = {
            'Price': self.data[self.close_col].dropna(),
            'Returns': self.analysis_results['returns'].dropna()
        }
        
        stationarity_results = {}
        
        for name, series in series_to_test.items():

            adf_result = adfuller(series)
            
            try:
                kpss_result = kpss(series, regression='c')
            except Exception as e:
                raise ValueError(f'Error: {e}')
            
            stationarity_results[name] = {
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'adf_stationary': adf_result[1] < 0.05,
                'kpss_statistic': kpss_result[0] if 'kpss_result' in locals() else None,
                'kpss_pvalue': kpss_result[1] if 'kpss_result' in locals() else None,
                'kpss_stationary': kpss_result[1] > 0.05 if 'kpss_result' in locals() else None
            }
        
        self.parameters['stationarity_tests'] = stationarity_results
        
        return self.analysis_results
    
    def distribution_analysis(
        self,
    ):
        
        """
        Distribution Analysis
        
        """
        
        returns = self.analysis_results['returns'].dropna()
        
        jb_stat, jb_pval = stats.jarque_bera(returns)
        shapiro_stat, shapiro_pval = stats.shapiro(returns)
        
        extreme_threshold = returns.std() * 3
        extreme_returns = returns[np.abs(returns) > extreme_threshold]
        
        distribution_results = {
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pval': jb_pval,
            'shapiro_stat': shapiro_stat,
            'shapiro_pval': shapiro_pval,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'excess_kurtosis': returns.kurtosis() - 3,
            'extreme_returns_count': len(extreme_returns),
            'extreme_returns_pct': len(extreme_returns)/len(returns)*100
        }
        
        self.parameters['distribution_analysis'] = distribution_results
        
        return self.analysis_results
    
    def autocorrelation_analysis(
        self,
        lags: List[int] = [5, 10, 20],
    ):
        
        """
        Autocorrelation Analysis
        
        Parameters:
        lags (List[int]): List of lags for autocorrelation analysis
        
        """

        returns = self.analysis_results['returns'].dropna()
        
        lb_test = acorr_ljungbox(returns, lags=lags, return_df=True)
        for lag in lags:
            pval = lb_test.loc[lb_test['lb_stat'].index == lag, 'lb_pvalue'].values[0]
        
        squared_returns = returns ** 2
        lb_test_squared = acorr_ljungbox(squared_returns, lags=lags, return_df=True)
        for lag in lags:
            pval = lb_test_squared.loc[lb_test_squared['lb_stat'].index == lag, 'lb_pvalue'].values[0]
        
        autocorrelation_results = {
            'returns_lb': lb_test,
            'squared_returns_lb': lb_test_squared,
            'lags': lags
        }
        
        self.parameters['autocorrelation_analysis'] = autocorrelation_results
        
        return self.analysis_results
    
    def volatility_analysis(
        self,
        window: int = 20,
    ):
        
        """
        Volatility Analysis
        
        Parameters:
        window (int): Window for rolling volatility calculation
        
        """

        returns = self.analysis_results['returns'].dropna()
        
        rolling_vol = returns.rolling(window=window).std()
        
        high_vol_threshold = rolling_vol.quantile(0.75)
        low_vol_threshold = rolling_vol.quantile(0.25)
        
        high_vol_periods = (rolling_vol > high_vol_threshold).sum()
        low_vol_periods = (rolling_vol < low_vol_threshold).sum()
        
        vol_autocorr = rolling_vol.autocorr(lag=1)
        
        volatility_results = {
            'rolling_volatility': rolling_vol,
            'high_vol_periods': high_vol_periods,
            'low_vol_periods': low_vol_periods,
            'avg_volatility': rolling_vol.mean(),
            'vol_of_vol': rolling_vol.std(),
            'volatility_persistence': vol_autocorr,
            'window': window
        }
        
        self.parameters['volatility_analysis'] = volatility_results
        
        return self.analysis_results
    
    def drawdown_analysis(
        self,
    ):
        
        """
        Drawdown Analysis
        
        """
        
        cumulative_returns = (1 + self.analysis_results['returns']).cumprod()
        
        running_max = cumulative_returns.expanding().max()
        
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin()
        avg_drawdown = drawdown.mean()
        
        drawdown_results = {
            'drawdown_series': drawdown,
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_drawdown_date,
            'avg_drawdown': avg_drawdown
        }
        
        self.parameters['drawdown_analysis'] = drawdown_results
        
        return self.analysis_results
    
    def seasonal_decomposition(
        self,
        period: int = 24,
    ):
        
        """
        Seasonal Decomposition
        
        Parameters:
        period (int): Period for seasonal decomposition
        
        """

        returns = self.analysis_results['returns'].dropna()
        
        if len(returns) > period * 2:
            try:
                decomposition = seasonal_decompose(returns, model='additive', period=period)
                
                seasonal_strength = decomposition.seasonal.std() / returns.std()
                residual_strength = decomposition.resid.std() / returns.std()
                
                decomposition_results = {
                    'decomposition': decomposition,
                    'seasonal_strength': seasonal_strength,
                    'residual_strength': residual_strength,
                    'trend_strength': 1 - seasonal_strength - residual_strength,
                    'period': period
                }
                
                self.parameters['seasonal_decomposition'] = decomposition_results
                
            except Exception as e:
                print(f"  Decomposition failed: {e}")
        else:
            print("  Insufficient data for seasonal decomposition")
        
        return self.analysis_results
    
    def run_complete_analysis(
        self,
        autocorrelation_lags: List[int] = [5, 10, 20],
        volatility_window: int = 20,
        seasonal_period: int = 24,
    ):
        
        """
        Run Complete Time Series Analysis
        
        Parameters:
        autocorrelation_lags (List[int]): Lags for autocorrelation analysis
        volatility_window (int): Window for volatility analysis
        seasonal_period (int): Period for seasonal decomposition
        
        """
        
        self.basic_descriptive_stats(prints = False)
        self.stationarity_tests()
        self.distribution_analysis()
        self.autocorrelation_analysis(lags=autocorrelation_lags)
        self.volatility_analysis(window=volatility_window)
        self.drawdown_analysis()
        self.seasonal_decomposition(period=seasonal_period)
        
        count_removed_rows = self.data.shape[0] - self.analysis_results.shape[0]
        
        print('='*50)
        print(self.analysis_results.info())
        print('='*50)
        print(f'Shape of analysis results {self.analysis_results.shape}')
        print('='*50)
        print(f'{count_removed_rows} rows removed')
        print('='*50)

        return self.analysis_results, self.parameters