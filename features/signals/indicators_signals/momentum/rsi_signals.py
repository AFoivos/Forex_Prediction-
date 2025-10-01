import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexRSISignals:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
    ):
        
        """
        Class for RSI signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
        print("="*50)
        print("RSI SIGNAL GENERATION")
        print("="*50)
        print("Available functions: \n1 rsi_overbought_oversold_signals \n2 rsi_centerline_signals \n3 rsi_divergence_signals \n4 rsi_momentum_signals \n5 rsi_failure_swing_signals \n6 rsi_trend_reversal_signals \n7 generate_all_rsi_signals")
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
            
    def rsi_overbought_oversold_signals(
        self,
        overbought: int = 70,
        oversold: int = 30,
        columns: List[str] = ['rsi_14', 'rsi_21', 'rsi_28'],
    ):
        
        """
        RSI Overbought/Oversold Signals
        2 = Overbought (RSI > 70)
        1 = Oversold (RSI < 30)
        0 = Normal (30 <= RSI <= 70)
        
        Parameters:     
        
        overbought (int): Overbought threshold
        oversold (int): Oversold threshold

        """

        self._validate_columns(columns = columns)
        
        for col in columns:
            overbought_condition = self.data[col] > overbought
            oversold_condition = self.data[col] < oversold
            
            self.signals[f'{col}_overbought'] = np.select(
                [overbought_condition, oversold_condition],
                [2,1],
                default=0
            )
            
        return self.signals
        
    def rsi_centerline_signals(
        self,
        columns: List[str] = ['rsi_14', 'rsi_21', 'rsi_28']
    ):
        
        """
        RSI Centerline (50) Crossover Signals
        2 = Bullish (RSI crosses above 50)
        1 = Bearish (RSI crosses below 50)
        0 = No crossover
        
        """
        
        self._validate_columns(columns = columns)
        
        for col in columns:
            
            bullish_centerline = (
                (self.data[col] > 50) & 
                (self.data[col].shift(1) <= 50)
            )

            bearish_centerline = (
                (self.data[col] < 50) & 
                (self.data[col].shift(1) >= 50)
            )
        
            self.signals[f"{col}_centerline"] = np.select(
                [bullish_centerline, bearish_centerline],
                [2, 1],
                default=0
            )
            
        return self.signals
    
    def rsi_divergence_signals(
        self,
        lookback: int = 4,
        columns: List[str] = ['rsi_14', 'rsi_21', 'rsi_28']
    ):
        
        """
        RSI Divergence Signals (Price vs RSI)
        2 = Bullish Divergence (Price lower low, RSI higher low)
        1 = Bearish Divergence (Price higher high, RSI lower high)
        0 = No divergence
        
        Parameters:
        lookback (int): Lookback period for RSI
        columns (list[str]): List of column names for RSI signals
        
        """
        
        self._validate_columns(columns = columns)

        for col in columns:
            
            # Bullish Divergence: Price makes lower low, RSI makes higher low
            price_lower_low = (
                (self.data[self.close_col] < self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) < self.data[self.close_col].shift(lookback + 1))
            )
            
            rsi_higher_low = (
                (self.data[col] > self.data[col].shift(lookback)) &
                (self.data[col].shift(1) > self.data[col].shift(lookback + 1))
            )
            
            bullish_divergence = price_lower_low & rsi_higher_low
            
            # Bearish Divergence: Price makes higher high, RSI makes lower high
            price_higher_high = (
                (self.data[self.close_col] > self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) > self.data[self.close_col].shift(lookback + 1))
            )
            
            rsi_lower_high = (
                (self.data[col] < self.data[col].shift(lookback)) &
                (self.data[col].shift(1) < self.data[col].shift(lookback + 1))
            )
            
            bearish_divergence = price_higher_high & rsi_lower_high
            
            self.signals[f"{col}_divergence"] = np.select(
                [bullish_divergence, bearish_divergence],
                [2, 1],
                default = 0
            )
    
        return self.signals
    
    def rsi_momentum_signals(
        self,
        columns: List[str] = ['rsi_14', 'rsi_21', 'rsi_28']
    ):
        
        """
        RSI Momentum Signals
        2 = RSI increasing (bullish momentum)
        1 = RSI decreasing (bearish momentum)
        0 = RSI stable
        
        Parameters:
        
        columns (list[str]): List of column names for RSI signals
        
        """
        
        self._validate_columns(columns = columns)
        
        for col in columns:
            
            slope_col = f"{col}_slope"
            self._validate_columns(columns = [slope_col])
            
            rsi_increasing = self.data[slope_col] > 0
            rsi_decreasing = self.data[slope_col] < 0
            
            self.signals[f"{col}_momentum"] = np.select(
                [rsi_increasing, rsi_decreasing],
                [2, 1],
                default = 0
            )
        
        return self.signals
    
    def rsi_failure_swing_signals(
        self,
        columns: List[str] = ['rsi_14', 'rsi_21', 'rsi_28'],
        oversold: int = 30,
        overbought: int = 70
    ):
        
        """
        RSI Failure Swing Signals (Classic RSI pattern)
        2 = Bullish Failure Swing (RSI oversold, bounces, then retests but stays above previous low)
        1 = Bearish Failure Swing (RSI overbought, pulls back, then retests but stays below previous high)
        0 = No failure swing
        
        Parameters:
        columns (list[str]): List of column names for RSI signals
        overbought (int): Overbought threshold
        oversold (int): Oversold threshold
        
        """
        
        self._validate_columns(columns)

        for col in columns:
            
            # Bullish Failure
            rsi_oversold = self.data[col].shift(2) < oversold
            rsi_bounce = self.data[col].shift(1) > self.data[col].shift(2)
            rsi_retest_higher_low = (
                (self.data[col] > oversold) &
                (self.data[col] > self.data[col].shift(2))
            )
            
            bullish_faillure_swing = rsi_oversold & rsi_bounce & rsi_retest_higher_low
            
            # Bearish Faillure Swing
            rsi_overbought = self.data[col].shift(2) > overbought
            rsi_pullback = self.data[col].shift(1) < self.data[col].shift(2)
            rsi_retest_lower_high = (
                (self.data[col] < overbought) &
                (self.data[col] < self.data[col].shift(2))
            )
            
            bearish_faillure_swing = rsi_overbought & rsi_pullback & rsi_retest_lower_high
            
            self.signals[f"{col}_swing_fail"] = np.select(
                [bullish_faillure_swing, bearish_faillure_swing],
                [2,1],
                default = 0
            )
            
        return self.signals
            
    def rsi_trend_reversal_signals(
        self,
        columns: List[str] = ['rsi_14', 'rsi_21', 'rsi_28'],
        oversold: int = 30,
        overbought: int = 70
    ):
        
        """
        RSI Trend Reversal Signals (Exit from overbought/oversold)
        2 = Bullish Reversal (RSI exits oversold <30 to >30)
        1 = Bearish Reversal (RSI exits overbought >70 to <70)
        0 = No reversal signal
        
        Parameters:
        columns (list[str]): List of column names for RSI signals
        overbought (int): Overbought threshold
        oversold (int): Oversold threshold
        
        """
        
        self._validate_columns(columns = columns)
        
        for col in columns:
            
            bullish_reversal = (
                (self.data[col] > oversold) &
                (self.data[col].shift(1) <= oversold)
            )
            
            bearish_reversal = (
                (self.data[col] < overbought) &
                (self.data[col].shift(1) >= overbought)
            )
            
            self.signals[f"{col}_reversal"] = np.select(
                [bullish_reversal, bearish_reversal],
                [2, 1],
                default = 0
            )
        
        return self.signals
    
    def generate_all_rsi_signals(self): # add parameters
        
        """
        Generate all RSI signals
        
        """
        
        self.rsi_overbought_oversold_signals()
        self.rsi_centerline_signals()
        self.rsi_divergence_signals()
        self.rsi_momentum_signals()
        self.rsi_failure_swing_signals()
        self.rsi_trend_reversal_signals()
        print(self.signals.tail(10), "\n", self.signals.shape)
        return self.signals