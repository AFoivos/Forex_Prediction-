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
    
    def add_volume_confirmation(self):
        
        """
        Adding volume confirmation features
        
        """
        
        print("="*50)
        print("VOLUME CONFIRMATION")
        print("="*50)
        
        added_columns = []
        
        # Volume-Price confirmation
        if all(col in self.data.columns for col in ['obv', 'volume_sma_20_ratio']):
            self.data['volume_price_confirmation'] = (
                (self.data[self.close_col].diff() > 0) & (self.data['obv_trend'] > 0) |
                (self.data[self.close_col].diff() < 0) & (self.data['obv_trend'] < 0)
            ).astype(int)
            
            self.data['volume_divergence'] = (
                (self.data[self.close_col].diff() > 0) & (self.data['obv_trend'] < 0) |
                (self.data[self.close_col].diff() < 0) & (self.data['obv_trend'] > 0)
            ).astype(int)
            
            added_columns.extend(['volume_price_confirmation', 'volume_divergence'])
        
        # High volume breakout detection
        if 'volume_sma_20_ratio' in self.data.columns:
            self.data['high_volume_breakout'] = (
                (self.data['volume_sma_20_ratio'] > 2.0) &
                (self.data[self.close_col].diff() > 0)
            ).astype(int)
            
            self.data['high_volume_breakdown'] = (
                (self.data['volume_sma_20_ratio'] > 2.0) &
                (self.data[self.close_col].diff() < 0)
            ).astype(int)
            
            added_columns.extend(['high_volume_breakout', 'high_volume_breakdown'])
        
        # Volume spike detection
        if 'volume_roc_5' in self.data.columns:
            self.data['volume_spike'] = (self.data['volume_roc_5'] > 100).astype(int)
            self.data['volume_drought'] = (self.data['volume_roc_5'] < -50).astype(int)
            
            added_columns.extend(['volume_spike', 'volume_drought'])
        
        print(f'New columns added: {added_columns}')
        print("="*50)
        
        return self.data
    
    def get_volume_score(self) -> Optional[float]:
        
        """
        Returns overall volume score
        
        100: Extremely High Volume Activity
        50: Normal Volume Activity
        0: Extremely Low Volume Activity
        
        """
        
        print("="*50)
        print("VOLUME SCORE")
        print("="*50)
        
        required_indicators = ['obv_trend', 'volume_sma_20_ratio', 'volume_roc_5']
        missing_indicators = [ind for ind in required_indicators if ind not in self.data.columns]
        
        if missing_indicators:
            print(f'Missing indicators: {missing_indicators}')
            print('Calculate OBV, Volume SMA and Volume ROC first')
            return None
        
        volume_score = 50.0  # Normal starting point
        
        # OBV trend contribution
        obv_contribution = np.clip(self.data['obv_trend'].iloc[-1] / abs(self.data['obv_trend'].iloc[-1] + 1e-10) * 10, -25, 25)
        volume_score += obv_contribution
        
        # Volume SMA ratio contribution
        volume_ratio = self.data['volume_sma_20_ratio'].iloc[-1]
        volume_ratio_contribution = np.clip((volume_ratio - 1.0) * 25, -25, 25)
        volume_score += volume_ratio_contribution
        
        # Volume ROC contribution
        volume_roc = self.data['volume_roc_5'].iloc[-1]
        volume_roc_contribution = np.clip(volume_roc / 4, -25, 25)
        volume_score += volume_roc_contribution
        
        # Ensure final score is between 0-100
        volume_score = np.clip(volume_score, 0, 100)
        volume_score = round(volume_score, 2)
        
        print(f"Current volume score: {volume_score}")
        
        # Interpret the score
        if volume_score >= 75:
            print("Extremely High Volume Activity")
        elif volume_score >= 60:
            print("High Volume Activity")
        elif volume_score >= 40:
            print("Normal Volume Activity")
        elif volume_score >= 25:
            print("Low Volume Activity")
        else:
            print("Extremely Low Volume Activity")
        
        return volume_score
    
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
        self.add_volume_confirmation()
        self.get_volume_score()
        
        return self.data