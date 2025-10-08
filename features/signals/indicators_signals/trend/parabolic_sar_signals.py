import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexParabolicSARSignals:  
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
        parameters: List = None, 
        prints = True
    ):
        
        """
        Class for Parabolic SAR signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        parameters (List): List of parameters for SAR [acceleration, maximum] or nested list
        
        """
        
        self.prints = prints
        
        if self.prints:
            print("="*50)
            print("PARABOLIC SAR SIGNAL GENERATION")
            print("="*50)
            print(" Available Functions: \n1. sar_trend_signals \n2. sar_reversal_signals \n3. sar_distance_signals \n4. sar_slope_signals \n5. sar_trend_confirmation_signals\n6. generate_all_sar_signals")
            print("="*50)   
        
        self.close_col = close_col
        self.data = data.copy() 
        
        self.signals = pd.DataFrame(
            {self.close_col: self.data[self.close_col]}, 
            index=self.data.index
        )
        
        self.sar_names = []
        self.sar_slope_names = []
        
        self.parameters = parameters
        if parameters is None:
            self.parameters = [[0.02, 0.2]] 
        
        self._validate_columns()
        self._extract_column_names()
    
    def _validate_columns(
        self,
        columns: list[str] = None,
    ):
        
        """
        Validate that required columns exist
        
        """
        
        required_cols = [self.close_col]
        
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
        self
    ):
        
        """
        Extract SAR column names based on parameters
        
        """
        
        is_nested = self._is_nested_list(self.parameters)
        
        if is_nested:
            for lst in self.parameters:
                sar_name = f'sar_{lst[0]}_{lst[1]}'
                slope_name = f'{sar_name}_slope'
                
                self._validate_columns([sar_name, slope_name])
                
                self.sar_names.append(sar_name)
                self.sar_slope_names.append(slope_name)
        else:
            sar_name = f'sar_{self.parameters[0]}_{self.parameters[1]}'
            slope_name = f'{sar_name}_slope'
            
            self._validate_columns([sar_name, slope_name])
            
            self.sar_names.append(sar_name)
            self.sar_slope_names.append(slope_name)
        
    def sar_trend_signals(
        self,
    ):
        
        """
        Parabolic SAR Trend Signals
        2 = Uptrend (Price above SAR)
        1 = Downtrend (Price below SAR)
        0 = No clear trend

        """
        for sar_name in self.sar_names:
            self._validate_columns([sar_name])
            
            column = sar_name   
        
            uptrend = self.data[self.close_col] > self.data[column]
            downtrend = self.data[self.close_col] < self.data[column]
            
            self.signals[f'{column}_trend'] = np.select(
                [uptrend, downtrend],
                [2, 1],
                default=0
            )
        
        return self.signals
    
    def sar_reversal_signals(
        self,
 
    ):
        
        """
        Parabolic SAR Reversal Signals
        2 = Bullish Reversal (SAR moves from above to below price)
        1 = Bearish Reversal (SAR moves from below to above price)
        0 = No reversal
        
        """
        
        for sar_name in self.sar_names:
            self._validate_columns([sar_name])
            
            column = sar_name
            
            self._validate_columns([column])
            
            bullish_reversal = (
                (self.data[self.close_col] > self.data[column]) & 
                (self.data[self.close_col].shift(1) <= self.data[column].shift(1))
            )
            
            bearish_reversal = (
                (self.data[self.close_col] < self.data[column]) & 
                (self.data[self.close_col].shift(1) >= self.data[column].shift(1))
            )
            
            self.signals[f'{column}_reversal'] = np.select(
                [bullish_reversal, bearish_reversal],
                [2, 1],
                default=0
            )
        
        return self.signals
    
    def sar_distance_signals(
        self,
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
        for sar_name in self.sar_names:
            self._validate_columns([sar_name])
            
            column = sar_name
            
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
            
            self.signals[f'{column}_distance'] = np.select(
                [strong_uptrend, strong_downtrend],
                [2, 1],
                default=0
            )
        
        return self.signals
    
    def sar_slope_signals(
        self,
    ):
        
        """
        Parabolic SAR Slope Signals (SAR acceleration)
        2 = SAR accelerating upward (downtrend strengthening)
        1 = SAR accelerating downward (uptrend strengthening)
        0 = SAR stable

        Parameters:
        column (str): Column name for Parabolic SAR
        
        """
        
        for sar_name in self.sar_names:
            self._validate_columns([f'{sar_name}_slope'])
            
            column = f'{sar_name}_slope'
        
            # Note: SAR moves opposite to price trend
            sar_accelerating_up = self.data[column] > 0  
            sar_accelerating_down = self.data[column] < 0  
            
            self.signals[f'{column}_signal'] = np.select(
                [sar_accelerating_up, sar_accelerating_down],
                [2, 1],
                default=0
            )
        
        return self.signals
    
    def sar_trend_confirmation_signals(
        self,
    ):
        
        """
        SAR Trend Confirmation Signals
        2 = SAR confirms uptrend (price above SAR & SAR below price accelerating)
        1 = SAR confirms downtrend (price below SAR & SAR above price accelerating)
        0 = No clear confirmation
        
        Parameters:
        columns (list[str]): List of column names for Parabolic SAR and slope
        
        """
        
        for sar_name in self.sar_names:
            self._validate_columns([sar_name, f'{sar_name}_slope'])
            
            sar = sar_name
            slope = f'{sar}_slope'
            
            
            confirms_uptrend = (
                (self.data[self.close_col] > self.data[sar]) & 
                (self.data[slope] < 0)  
            )
            
            confirms_downtrend = (
                (self.data[self.close_col] < self.data[sar]) & 
                (self.data[slope] > 0)  
            )
            
            self.signals[f'{sar}_trend_confirmation'] = np.select(
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
        
        count_removed_rows = self.data.shape[0] - self.signals.shape[0]
        
        if self.prints:
            print('='*50)
            print('Data Info')
            print(self.signals.info())
            print('='*50)
            print(f'Shape of data {self.signals.shape}')
            print('='*50)
            print(f'{count_removed_rows} rows removed')
            print('='*50)
        
        return self.signals