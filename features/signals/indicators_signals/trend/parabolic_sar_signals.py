import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexParabolicSARSiganls:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
    ):
        
        """
        Class for Parabolic SAR signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
        print("="*50)
        print("PARABOLIC SAR SIGNAL GENERATION")
        print("="*50)
        print(" Available Fuctions: \n1 sar_trend_signals \n2 sar_reversal_signals \n3 sar_distance_signals \n4 sar_slope_signals \n5 sar_trend_confirmation_signals\n6 generate_all_sar_signals")
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
        
    def sar_trend_signals(
        self,
        column: str = 'trend_parabolic_sar',
    ):
        
        """
        Parabolic SAR Trend Signals
        2 = Uptrend (Price above SAR)
        1 = Downtrend (Price below SAR)
        0 = No clear trend
        
        Parameters:
        column (str): Column name for Parabolic SAR

        """
        
        self._validate_columns([column])
        
        uptrend = self.data[self.close_col] > self.data[column]
        downtrend = self.data[self.close_col] < self.data[column]
        
        self.signals['sar_trend'] = np.select(
            [uptrend, downtrend],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def sar_reversal_signals(
        self,
        column: str = 'trend_parabolic_sar',
    ):
        
        """
        Parabolic SAR Reversal Signals
        2 = Bullish Reversal (SAR moves from above to below price)
        1 = Bearish Reversal (SAR moves from below to above price)
        0 = No reversal
        
        """
        
        self._validate_columns([column])
        
        bullish_reversal = (
            (self.data[self.close_col] > self.data[column]) & 
            (self.data[self.close_col].shift(1) <= self.data[column].shift(1))
        )
        
        bearish_reversal = (
            (self.data[self.close_col] < self.data[column]) & 
            (self.data[self.close_col].shift(1) >= self.data[column].shift(1))
        )
        
        self.signals['sar_reversal'] = np.select(
            [bullish_reversal, bearish_reversal],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def sar_distance_signals(
        self,
        column: str = 'trend_parabolic_sar', 
        threshold=0.005
    ):
        
        """
        Parabolic SAR Distance Signals
        2 = Price significantly above SAR (strong uptrend)
        1 = Price significantly below SAR (strong downtrend)
        0 = Price close to SAR (consolidation)
        
        Parameters:
        column (str): Column name for Parabolic SAR
        threshold (float): Threshold for distance from SAR
        
        """
        
        self._validate_columns([column])
        
        # Calculate percentage distance from SAR
        distance_pct = abs(self.data[self.close_col] - self.data[column]) / self.data[self.close_col]
        
        strong_uptrend = (
            (self.data[self.close_col] > self.data[column]) & 
            (distance_pct > threshold)
        )
        
        strong_downtrend = (
            (self.data[self.close_col] < self.data[column]) & 
            (distance_pct > threshold)
        )
        
        self.signals['sar_distance'] = np.select(
            [strong_uptrend, strong_downtrend],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def sar_slope_signals(
        self,
        column: str = 'trend_parabolic_sar_slope'
    ):
        
        """
        Parabolic SAR Slope Signals (SAR acceleration)
        2 = SAR accelerating upward (downtrend strengthening)
        1 = SAR accelerating downward (uptrend strengthening)
        0 = SAR stable

        Parameters:
        column (str): Column name for Parabolic SAR
        
        """
        
        self._validate_columns([column])
        
        # Note: SAR moves opposite to price trend
        sar_accelerating_up = self.data[column] > 0  
        sar_accelerating_down = self.data[column] < 0  
        
        self.signals['sar_slope_signal'] = np.select(
            [sar_accelerating_up, sar_accelerating_down],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def sar_trend_confirmation_signals(
        self,
        columns: List[str] = ['trend_parabolic_sar', 'trend_parabolic_sar_slope']
    ):
        
        """
        SAR Trend Confirmation Signals
        2 = SAR confirms uptrend (price above SAR & SAR below price accelerating)
        1 = SAR confirms downtrend (price below SAR & SAR above price accelerating)
        0 = No clear confirmation
        
        Parameters:
        columns (list[str]): List of column names for Parabolic SAR and slope
        
        """
        
        self._validate_columns(columns)
        
        confirms_uptrend = (
            (self.data[self.close_col] > self.data[columns[0]]) & 
            (self.data[columns[1]] < 0)  
        )
        
        confirms_downtrend = (
            (self.data[self.close_col] < self.data[columns[0]]) & 
            (self.data[columns[1]] > 0)  
        )
        
        self.signals['sar_trend_confirmation'] = np.select(
            [confirms_uptrend, confirms_downtrend],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def generate_all_sar_signals(self): #add parameters
        
        """
        Generate all Parabolic SAR signals
        
        """
        
        self.sar_trend_signals()
        self.sar_reversal_signals()
        self.sar_distance_signals()
        self.sar_slope_signals()
        self.sar_trend_confirmation_signals()
        print(self.signals.tail(10), "\n", self.signals.shape)
        return self.signals