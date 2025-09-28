import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexADXSignals:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
    ):
        """
        Class for ADX signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
        print("="*50)
        print("ADX SIGNAL GENERATION")
        print("="*50)
        
        self.close_col = close_col
        self.data = data.copy()
        
        self.signals = pd.DataFrame(
            {self.close_col: self.data[self.close_col]},
            index=self.data.index
        )
        
        self._validate_columns()
    
    def _validate_columns(
        self, 
        columns: list[str] = None,
    ): 
        
        """
        Validate that required indicator columns exist
            
        """
        
        required_cols = [
            self.close_col,
        ]
        
        if columns is not None:
            required_cols.extend(columns)
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        
    def adx_trend_strength_signals(
        self, 
        strong_threshold: int = 25, 
        weak_threshold: int = 20,
        column: str = 'trend_adx',
    ):
        
        """
        ADX Trend Strength Signals
        2 = Strong Trend (ADX > 25)
        1 = Weak Trend (ADX < 20) 
        0 = Developing Trend (20 <= ADX <= 25)
        
        """
        
        self._validate_columns([column]) # Validate the ADX column
        
        strong_trend = self.data[column] > strong_threshold
        weak_trend = self.data[column] < weak_threshold
        
        self.signals['adx_trend_strength'] = np.select(
            [strong_trend, weak_trend],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def adx_direction_signals(
        self,
        columns: list[str] = ['trend_plus_di', 'trend_minus_di'],
    ):
        
        """
        ADX Direction Signals (using +DI and -DI)
        2 = Bullish (+DI > -DI)
        1 = Bearish (-DI > +DI)
        0 = Neutral
        
        """
        
        self._validate_columns(columns)
        
        bullish = self.data[columns[0]] > self.data[columns[1]]
        bearish = self.data[columns[1]] > self.data[columns[0]]
        
        self.signals['adx_direction'] = np.select(
            [bullish, bearish],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def adx_di_crossover_signals(
        self,
        columns: list[str] = ['trend_plus_di', 'trend_minus_di'],
    ):
        
        """
        +DI/-DI Crossover Signals
        2 = +DI crosses above -DI (Bullish)
        1 = -DI crosses above +DI (Bearish)
        0 = No crossover
        
        """
        
        self._validate_columns(columns)
        
        bullish_cross = (
            (self.data[columns[0]] > self.data[columns[1]]) & 
            (self.data[columns[0]].shift(1) <= self.data[columns[1]].shift(1))
        )
        
        bearish_cross = (
            (self.data[columns[1]] > self.data[columns[0]]) & 
            (self.data[columns[1]].shift(1) <= self.data[columns[0]].shift(1))
        )
        
        self.signals['adx_di_crossover'] = np.select(
            [bullish_cross, bearish_cross],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def adx_slope_signals(
        self,
        column: str = 'trend_adx_slope',
    ):
        
        """
        ADX Slope Signals (Momentum of trend strength)
        2 = ADX increasing (trend strengthening)
        1 = ADX decreasing (trend weakening)
        0 = ADX flat
        
        """
        
        self._validate_columns([column])
        
        increasing = self.data[column] > 0
        decreasing = self.data[column] < 0
        
        self.signals['adx_slope'] = np.select(
            [increasing, decreasing],
            [2, 1],
            default=0
        )

        return self.signals
    
    def adx_comprehensive_signals(
        self,
        strong_threshold: int =25,
        columns: list[str] = ['trend_adx', 'trend_adx_pos', 'trend_adx_neg'],
    ):
        
        """
        Comprehensive ADX Signals (Trend Strength + Direction)
        2 = Strong Bullish Trend (ADX > 25 & +DI > -DI)
        1 = Strong Bearish Trend (ADX > 25 & -DI > +DI)
        0 = Weak/No clear trend
        
        """
        
        
        self._validate_columns(columns)
        
        strong_bullish = (
            (self.data[columns[0]] > strong_threshold) & 
            (self.data[columns[1]] > self.data[columns[2]])
        )
        
        strong_bearish = (
            (self.data[columns[0]] > strong_threshold) & 
            (self.data[columns[2]] > self.data[columns[1]])
        )
        
        self.signals['adx_comprehensive'] = np.select(
            [strong_bullish, strong_bearish],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def generate_all_adx_signals(self):#add parameters
        
        """
        Generate all ADX signals
        
        """
        
        self.adx_trend_strength_signals()
        self.adx_direction_signals()
        self.adx_di_crossover_signals()
        self.adx_slope_signals()
        self.adx_comprehensive_signals()
        print(self.signals.tail(10), "\n", self.signals.shape)
        return self.signals