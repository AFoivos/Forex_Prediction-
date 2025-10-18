import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

from itertools import product

import warnings
warnings.filterwarnings('ignore')

class ForexSTDSignals:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
        parameters: List = None,
        prints = True
    ):
        
        """
        Class for Standard Deviation signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
        self.prints = prints
        
        if self.prints:
            print("="*50)
            print("STANDARD DEVIATION SIGNAL GENERATION")
            print("="*50)
            print("Available functions: \n1 std_volatility_regime_signals \n2 std_price_position_signals \n3 std_breakout_signals \n4 std_squeeze_signals \n5 std_trend_signals \n6 std_divergence_signals \n7 generate_all_std_signals")
            print("="*50)
        
        self.close_col = close_col
        self.data = data.copy()
        
        self.signals = {}
        signals_names= [
            'overbought_oversold',      # Υπερβολική αγορά/πώληση
            'reversal',                 # Ανατροπή
            'volatility',               # Μεταβλητότητα
            'breakout',                 # Εκρήξεις
            'divergence',               # Αποκλίσεις
            'squeeze',                  # Συμπίεση
        ]

        for name in signals_names:
            self.signals[name] = pd.DataFrame(
                {self.close_col: self.data[self.close_col]},
                index = self.data.index
            )
        
        self.std_columns = []
        
        self.parameters = [20, 50, 100] if parameters is None else parameters
        
        self._validate_columns()
        self._extract_column_names(parameters = parameters)
        
    def _validate_columns(
        self, 
        columns: list[str] = None,
    ):  
        
        """
        Validate that required indicator columns exist
        
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
        
        return all(isinstance(item, list) for item in lst)
    
    def _extract_column_names(
        self,
        parameters: List = None,
    ):
        
        """
        Extract Standard Deviation column names based on parameters
        
        """
        is_nested = self._is_nested_list(parameters)
        
        if is_nested:
            for sublist in parameters:
                for period in sublist:
                    col_name = f'std_dev_{period}'
                    self.std_columns.extend([col_name])
        else:
            for period in parameters:
                col_name = f'std_dev_{period}'
                self.std_columns.extend([col_name])
    
    def std_volatility_regime_signals(
        self,
        lookback_period: int = 20,
        high_vol_threshold: float = 1.5,
        low_vol_threshold: float = 0.5
    ):
        
        """
        Standard Deviation Volatility Regime Signals
        2 = High Volatility (STD > rolling mean * high_vol_threshold)
        1 = Low Volatility (STD < rolling mean * low_vol_threshold)
        0 = Normal Volatility
        
        Parameters:
        lookback_period (int): Period for rolling mean calculation
        high_vol_threshold (float): Threshold for high volatility
        low_vol_threshold (float): Threshold for low volatility
        
        """
        
        for std_col in self.std_columns:
            self._validate_columns(columns = [std_col])
            
            rolling_mean = self.data[std_col].rolling(window=lookback_period).mean()
            
            high_vol = self.data[std_col] > (rolling_mean * high_vol_threshold)
            low_vol = self.data[std_col] < (rolling_mean * low_vol_threshold)
            
            self.signals['volatility'][f'{std_col}_regime'] = np.select(
                [high_vol, low_vol],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def std_price_position_signals(
        self,
        std_multiplier: float = 2.0
    ):
        
        """
        Standard Deviation Price Position Signals
        2 = Price > mean + (STD * multiplier) (Overbought zone)
        1 = Price < mean - (STD * multiplier) (Oversold zone)
        0 = Price within normal range
        
        Parameters:
        std_multiplier (float): Number of standard deviations for zones
        
        """
        
        for std_col in self.std_columns:
            self._validate_columns(columns = [std_col, self.close_col])
            
            period = int(std_col.split('_')[-1])
            
            rolling_mean = self.data[self.close_col].rolling(window = period).mean()
            
            upper_band = rolling_mean + (self.data[std_col] * std_multiplier)
            lower_band = rolling_mean - (self.data[std_col] * std_multiplier)
            
            above_upper = self.data[self.close_col] > upper_band
            below_lower = self.data[self.close_col] < lower_band
            
            self.signals['overbought_oversold'][f'{std_col}_price_position'] = np.select(
                [above_upper, below_lower],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def std_breakout_signals(
        self,
        std_multiplier: float = 2.0
    ):
        
        """
        Standard Deviation Breakout Signals
        2 = Breakout above upper band
        1 = Breakdown below lower band
        0 = No breakout
        
        Parameters:
        std_multiplier (float): Number of standard deviations for bands
        
        """
        
        for std_col in self.std_columns:
            self._validate_columns(columns = [std_col, self.close_col])
            
            period = int(std_col.split('_')[-1])
            rolling_mean = self.data[self.close_col].rolling(window=period).mean()
            
            upper_band = rolling_mean + (self.data[std_col] * std_multiplier)
            lower_band = rolling_mean - (self.data[std_col] * std_multiplier)
            
            breakout_above = (
                (self.data[self.close_col] > upper_band) & 
                (self.data[self.close_col].shift(1) <= upper_band.shift(1))
            )
            
            breakdown_below = (
                (self.data[self.close_col] < lower_band) & 
                (self.data[self.close_col].shift(1) >= lower_band.shift(1))
            )
            
            self.signals['breakout'][f'{std_col}_breakout'] = np.select(
                [breakout_above, breakdown_below],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def std_squeeze_signals(
        self,
        squeeze_period: int = 20,
        squeeze_threshold: float = 0.1
    ):
        
        """
        Standard Deviation Squeeze Signals
        2 = Volatility Squeeze (STD at relative lows)
        1 = Volatility Expansion (STD at relative highs)
        0 = Normal
        
        Parameters:
        squeeze_period (int): Period for squeeze detection
        squeeze_threshold (float): Threshold for squeeze identification
        
        """
        
        for std_col in self.std_columns:
            self._validate_columns(columns = [std_col])
            
            std_min = self.data[std_col].rolling(window=squeeze_period).min()
            std_max = self.data[std_col].rolling(window=squeeze_period).max()
            
            std_position = (self.data[std_col] - std_min) / (std_max - std_min)
            std_position = std_position.replace([np.inf, -np.inf], 0.5).fillna(0.5)
            
            squeeze = std_position < squeeze_threshold
            expansion = std_position > (1 - squeeze_threshold)
            
            self.signals['squeeze'][f'{std_col}_squeeze'] = np.select(
                [squeeze, expansion],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def std_trend_signals(
        self,
        trend_period: int = 5
    ):
        
        """
        Standard Deviation Trend Signals
        2 = Volatility Increasing (STD trending up)
        1 = Volatility Decreasing (STD trending down)
        0 = Volatility Stable
        
        Parameters:
        trend_period (int): Period for trend calculation
        
        """
        
        for std_col in self.std_columns:
            self._validate_columns(columns = [std_col])
            
            std_slope = self.data[std_col].diff(periods = trend_period)
            
            increasing_vol = std_slope > 0
            decreasing_vol = std_slope < 0
            
            self.signals['volatility'][f'{std_col}_trend'] = np.select(
                [increasing_vol, decreasing_vol],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def std_divergence_signals(
        self,
        lookback: int = 10
    ):
        
        """
        Standard Deviation Divergence Signals
        2 = Bullish Divergence (Price Lower Low, STD Higher High)
        1 = Bearish Divergence (Price Higher High, STD Lower Low)
        0 = No Divergence
        
        Parameters:
        lookback (int): Lookback period for divergence detection
        
        """
        
        for std_col in self.std_columns:
            self._validate_columns(columns = [std_col, self.close_col])
            
            price_lower_low = (
                (self.data[self.close_col] < self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) < self.data[self.close_col].shift(lookback + 1))
            )
            std_higher_high = (
                (self.data[std_col] > self.data[std_col].shift(lookback)) &
                (self.data[std_col].shift(1) > self.data[std_col].shift(lookback + 1))
            )
            bullish_divergence = price_lower_low & std_higher_high
            
            price_higher_high = (
                (self.data[self.close_col] > self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) > self.data[self.close_col].shift(lookback + 1))
            )
            std_lower_low = (
                (self.data[std_col] < self.data[std_col].shift(lookback)) &
                (self.data[std_col].shift(1) < self.data[std_col].shift(lookback + 1))
            )
            bearish_divergence = price_higher_high & std_lower_low
            
            self.signals['divergence'][f'{std_col}_divergence'] = np.select(
                [bullish_divergence, bearish_divergence],
                [2, 1],
                default = 0
            )
       
        return self.signals
    
    def std_mean_reversion_signals(
        self,
        std_multiplier: float = 1.0
    ):
        
        """
        Standard Deviation Mean Reversion Signals
        2 = Price far above mean (potential mean reversion short)
        1 = Price far below mean (potential mean reversion long)
        0 = Price near mean
        
        Parameters:
        std_multiplier (float): Number of standard deviations for mean reversion
        
        """
        
        for std_col in self.std_columns:
            self._validate_columns(columns = [std_col, self.close_col])
            
            period = int(std_col.split('_')[-1])
            rolling_mean = self.data[self.close_col].rolling(window=period).mean()
            
            distance_from_mean = (self.data[self.close_col] - rolling_mean) / self.data[std_col]
            
            far_above = distance_from_mean > std_multiplier
            far_below = distance_from_mean < -std_multiplier
            
            self.signals['reversal'][f'{std_col}_mean_reversion'] = np.select(
                [far_above, far_below],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def std_volatility_cluster_signals(
        self,
        cluster_period: int = 5,
        high_vol_threshold: float = 1.2
    ):
        
        """
        Standard Deviation Volatility Cluster Signals
        2 = High Volatility Cluster (multiple high STD periods)
        1 = Low Volatility Cluster (multiple low STD periods)
        0 = Normal
        
        Parameters:
        cluster_period (int): Period for cluster detection
        high_vol_threshold (float): Threshold for high volatility
        
        """
        
        for std_col in self.std_columns:
            self._validate_columns(columns = [std_col])
            
            rolling_mean = self.data[std_col].rolling(window=20).mean()
            high_vol = self.data[std_col] > (rolling_mean * high_vol_threshold)
            low_vol = self.data[std_col] < (rolling_mean * 0.8)
            
            high_vol_cluster = high_vol.rolling(window=cluster_period).sum() >= (cluster_period - 1)
            low_vol_cluster = low_vol.rolling(window=cluster_period).sum() >= (cluster_period - 1)
            
            self.signals['volatility'][f'{std_col}_vol_cluster'] = np.select(
                [high_vol_cluster, low_vol_cluster],
                [2, 1],
                default = 0
            )
         
        return self.signals
        
    def generate_all_std_signals(
        self,
        lookback_period: int = 20,
        high_vol_threshold: float = 1.5,
        low_vol_threshold: float = 0.5,
        std_multiplier: float = 2.0,
        squeeze_period: int = 20,
        squeeze_threshold: float = 0.1,
        trend_period: int = 5,
        lookback: int = 10,
        cluster_period: int = 5
    ):
        
        """
        Generate all Standard Deviation signals
        
        """
        
        self.std_volatility_regime_signals(
            lookback_period = lookback_period,
            high_vol_threshold = high_vol_threshold,
            low_vol_threshold = low_vol_threshold
        )
        self.std_price_position_signals(std_multiplier = std_multiplier)
        self.std_breakout_signals(std_multiplier = std_multiplier)
        self.std_squeeze_signals(
            squeeze_period = squeeze_period,
            squeeze_threshold = squeeze_threshold
        )
        self.std_trend_signals(trend_period = trend_period)
        self.std_divergence_signals(lookback = lookback)
        self.std_mean_reversion_signals(std_multiplier = std_multiplier)
        self.std_volatility_cluster_signals(cluster_period = cluster_period)
        
        count_removed_rows = 0
        for name in self.signals.keys():
            count_removed_rows += self.signals[name].shape[0] - self.data.shape[0]
        
        if self.prints:
            print('='*50)
            for name in self.signals.keys():
                print(self.signals[name].info())
            print('='*50)   
            for name in self.signals.keys():
                print(f'Shape of {name} data {self.signals[name].shape}')
            print('='*50)   
            print(f'{count_removed_rows} rows removed')
            print('='*50)
        
        return self.signals