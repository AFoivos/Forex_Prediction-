import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexVolumeIndicators:
    def __init__(self, 
                 data: pd.DataFrame,
                 open_col: str = 'open',
                 high_col: str = 'high', 
                 low_col: str = 'low', 
                 close_col: str = 'close',
                 volume_col: str = 'volume'):
        
        """
        Class for Volume Indicators
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        open_col (str): Column name for open price
        high_col (str): Column name for high price
        low_col (str): Column name for low price
        close_col (str): Column name for close price
        volume_col (str): Column name for volume
        
        """
        
        print("="*50)
        print("VOLUME INDICATORS")
        print("="*50)
        print(" Available Fuctions \n1 add_obv \n2 add_volume_sma \n3 add_volume_roc \n4 add_volume_confirmation")
        print("="*50)
        
        self.data = data.copy()
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        
        # Validate data_cols
        required_cols = [self.open_col, self.high_col, self.low_col, self.close_col, self.volume_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    def add_obv(self):
        
        """
        On-Balance Volume
        
        """
        
        print("="*50)
        print("ON-BALANCE VOLUME (OBV)")
        print("="*50)
        
        if not self.has_volume:
            print("⏭️  Skipping OBV - Volume data not available")
            return self.data
        
        obv_columns = [
            'obv', 'obv_trend', 'obv_signal', 
            'obv_divergence', 'obv_ma_ratio'
        ]
        
        # Calculate OBV
        self.data['obv'] = talib.OBV(self.data[self.close_col], self.data[self.volume_col])
        
        # OBV signals and features
        self.data['obv_trend'] = self.data['obv'].diff()
        self.data['obv_signal'] = np.where(self.data['obv_trend'] > 0, 1, -1)
        
        # OBV divergence (price vs volume)
        price_change = self.data[self.close_col].diff()
        self.data['obv_divergence'] = np.where(
            (price_change > 0) & (self.data['obv_trend'] < 0) |
            (price_change < 0) & (self.data['obv_trend'] > 0), 1, 0
        )
        
        # OBV to moving average ratio
        if len(self.data) > 20:
            obv_ma = self.data['obv'].rolling(window=20).mean()
            self.data['obv_ma_ratio'] = self.data['obv'] / obv_ma
        
        print(f'New columns added: {added_columns}')
        print("="*50)
        
        return self.data
    
    def add_volume_sma(self, periods: List[int] = [5, 10, 20, 50]):
        
        """
        Volume Simple Moving Averages
        
        Parameters:
        periods (List[int]): List of periods for Volume SMA
        
        """
        
        print("="*50)
        print("VOLUME SMA")
        print("="*50)
        
        added_columns = []
        for period in periods:
            col_name = f'volume_sma_{period}'
            self.data[col_name] = self.data[self.volume_col].rolling(window=period).mean()
            
            # Volume SMA ratios and signals
            self.data[f'{col_name}_ratio'] = self.data[self.volume_col] / self.data[col_name]
            self.data[f'{col_name}_signal'] = np.where(
                self.data[self.volume_col] > self.data[col_name], 1, -1
            )
            self.data[f'{col_name}_trend'] = self.data[col_name].diff()
            
            added_columns.extend([col_name, f'{col_name}_ratio', 
                                f'{col_name}_signal', f'{col_name}_trend'])
        
        print(f'New columns added: {added_columns}')
        print("="*50)
        
        return self.data
    
    def add_volume_roc(self, periods: List[int] = [5, 10, 14, 21]):
        
        """
        Volume Rate of Change
        
        Parameters:
        periods (List[int]): List of periods for Volume ROC
        
        """
        
        print("="*50)
        print("VOLUME RATE OF CHANGE")
        print("="*50)
        
        added_columns = []
        
        for period in periods:
            col_name = f'volume_roc_{period}'
            
            # Calculate Volume ROC
            self.data[col_name] = (
                (self.data[self.volume_col] - self.data[self.volume_col].shift(period)) / 
                self.data[self.volume_col].shift(period)
            ) * 100
            
            # Volume ROC signals
            self.data[f'{col_name}_signal'] = np.where(self.data[col_name] > 0, 1, -1)
            self.data[f'{col_name}_high_vol'] = (self.data[col_name] > 50).astype(int)
            self.data[f'{col_name}_low_vol'] = (self.data[col_name] < -50).astype(int)
            self.data[f'{col_name}_trend'] = self.data[col_name].diff()
            
            added_columns.extend([col_name, f'{col_name}_signal', 
                                f'{col_name}_high_vol', f'{col_name}_low_vol',
                                f'{col_name}_trend'])
        
        print(f'New columns added: {added_columns}')
        print("="*50)
        
        return self.data
    
    def get_all_volume_indicators(self,
                                 volume_sma_periods: List[int] = [5, 10, 20, 50],
                                 volume_roc_periods: List[int] = [5, 10, 14, 21]):
        
        """
        Adds all volume indicators
        
        Parameters:
        volume_sma_periods (List[int]): List of periods for Volume SMA
        volume_roc_periods (List[int]): List of periods for Volume ROC
        
        """
        
        self.add_obv()
        self.add_volume_sma(volume_sma_periods)
        self.add_volume_roc(volume_roc_periods)
        # self.add_volume_confirmation()
        # self.get_volume_score()
        
        return self.data