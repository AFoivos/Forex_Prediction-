import pandas as pd
import numpy as np
import talib
from typing import List, Optional, Tuple

import warnings
warnings.filterwarnings('ignore')

class VolatilityIndicators:
    def __init__(self, 
                 data: pd.DataFrame,
                 open_col: str = 'open',
                 high_col: str = 'high', 
                 low_col: str = 'low', 
                 close_col: str = 'close',
                 volume_col: str = 'volume'):
        
        """
        Class for Volatility Indicators
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        open_col (str): Column name for open price
        high_col (str): Column name for high price
        low_col (str): Column name for low price
        close_col (str): Column name for close price
        volume_col (str): Column name for volume
        
        """
        
        print("="*50)
        print("VOLATILITY INDICATORS")
        print("="*50)
        print(" Available Fuctions \n1 add_atr \n2 add_bollinger_bands \n3 add_keltner_channels \n4 add_standard_deviation \n5 add_volatility_confirmation \n6 get_volatility_score \n7 get_all_volatility_indicators")
        print("="*50)
        
        self.data = data.copy()
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        
        # Validate data_cols
        required_cols = [self.open_col, self.high_col, self.low_col, self.close_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    def add_atr(self, periods: List[int] = [14, 21, 28]):
        
        """
        Average True Range
        
        Parameters:
        periods (List[int]): List of periods for ATR
        
        """
        
        print("="*50)
        print("ATR INDICATOR")
        print("="*50)
        
        added_columns = []
        for period in periods:
            col_name = f'atr_{period}'
            self.data[col_name] = talib.ATR(
                self.data[self.high_col],
                self.data[self.low_col],
                self.data[self.close_col],
                timeperiod=period
            )
            
            # ATR signals and ratios
            self.data[f'{col_name}_pct'] = (self.data[col_name] / self.data[self.close_col]) * 100
            self.data[f'{col_name}_trend'] = self.data[col_name].diff()
            self.data[f'{col_name}_high_vol'] = (self.data[f'{col_name}_pct'] > 2.0).astype(int)
            
            added_columns.extend([col_name, f'{col_name}_pct', 
                                f'{col_name}_trend', f'{col_name}_high_vol'])
        
        print(f'New columns added: {", ".join(added_columns)}')
        print("="*50)
        
        return self.data
    
    def add_bollinger_bands(self,
                           periods: List[int] = [20, 50],
                           nbdevup: float = 2.0,
                           nbdevdn: float = 2.0):
        
        """
        Bollinger Bands
        
        Parameters:
        periods (List[int]): List of periods for Bollinger Bands
        nbdevup (float): Number of standard deviations for upper band
        nbdevdn (float): Number of standard deviations for lower band
        
        """
        
        print("="*50)
        print("BOLLINGER BANDS")
        print("="*50)
        
        added_columns = []
        
        for period in periods:
            prefix = f'bb_{period}'
            
            upper, middle, lower = talib.BBANDS(
                self.data[self.close_col],
                timeperiod=period,
                nbdevup=nbdevup,
                nbdevdn=nbdevdn
            )
            
            self.data[f'{prefix}_upper'] = upper
            self.data[f'{prefix}_middle'] = middle
            self.data[f'{prefix}_lower'] = lower
            
            # Bollinger Band signals
            self.data[f'{prefix}_width'] = (upper - lower) / middle
            self.data[f'{prefix}_pct_b'] = (self.data[self.close_col] - lower) / (upper - lower) * 100
            
            # Price position relative to bands
            self.data[f'{prefix}_above_upper'] = (self.data[self.close_col] > upper).astype(int)
            self.data[f'{prefix}_below_lower'] = (self.data[self.close_col] < lower).astype(int)
            self.data[f'{prefix}_squeeze'] = (self.data[f'{prefix}_width'] < 0.1).astype(int)
            
            # Band crossover signals
            self.data[f'{prefix}_signal'] = np.where(
                self.data[self.close_col] > upper, -1,  # Overbought
                np.where(self.data[self.close_col] < lower, 1, 0)  # Oversold
            )
            
            added_columns.extend([
                f'{prefix}_upper', f'{prefix}_middle', f'{prefix}_lower',
                f'{prefix}_width', f'{prefix}_pct_b', f'{prefix}_above_upper',
                f'{prefix}_below_lower', f'{prefix}_squeeze', f'{prefix}_signal'
            ])
        
        print(f'New columns added: {", ".join(added_columns)}')
        print("="*50)
        
        return self.data
    
    def add_keltner_channels(self,
                            ema_period: int = 20,
                            atr_period: int = 10,
                            atr_multiplier: float = 2.0):
        
        """
        Keltner Channels
        
        Parameters:
        ema_period (int): Period for EMA
        atr_period (int): Period for ATR
        atr_multiplier (float): Multiplier for ATR
        
        """
        
        print("="*50)
        print("KELTNER CHANNELS")
        print("="*50)
        
        # Calculate EMA middle line
        ema = talib.EMA(self.data[self.close_col], timeperiod=ema_period)
        
        # Calculate ATR for channel width
        atr = talib.ATR(
            self.data[self.high_col],
            self.data[self.low_col],
            self.data[self.close_col],
            timeperiod=atr_period
        )
        
        keltner_columns = [
            'keltner_upper', 'keltner_middle', 'keltner_lower',
            'keltner_width', 'keltner_pct_b', 'keltner_signal'
        ]
        
        self.data['keltner_middle'] = ema
        self.data['keltner_upper'] = ema + (atr * atr_multiplier)
        self.data['keltner_lower'] = ema - (atr * atr_multiplier)
        
        # Keltner Channel signals
        self.data['keltner_width'] = self.data['keltner_upper'] - self.data['keltner_lower']
        self.data['keltner_pct_b'] = (self.data[self.close_col] - self.data['keltner_lower']) / \
                                   (self.data['keltner_upper'] - self.data['keltner_lower']) * 100
        
        self.data['keltner_signal'] = np.where(
            self.data[self.close_col] > self.data['keltner_upper'], -1,  # Overbought
            np.where(self.data[self.close_col] < self.data['keltner_lower'], 1, 0)  # Oversold
        )
        
        print(f'New columns added: {", ".join(keltner_columns)}')
        print("="*50)
        
        return self.data
    
    def add_standard_deviation(self, periods: List[int] = [20, 50, 100]):
        
        """
        Standard Deviation
        
        Parameters:
        periods (List[int]): List of periods for Standard Deviation
        
        """
        
        print("="*50)
        print("STANDARD DEVIATION")
        print("="*50)
        
        added_columns = []
        for period in periods:
            col_name = f'std_dev_{period}'
            self.data[col_name] = talib.STDDEV(
                self.data[self.close_col],
                timeperiod=period,
                nbdev=1
            )
            
            # Standard Deviation signals
            self.data[f'{col_name}_pct'] = (self.data[col_name] / self.data[self.close_col]) * 100
            self.data[f'{col_name}_zscore'] = (self.data[self.close_col] - self.data[self.close_col].rolling(period).mean()) / self.data[col_name]
            self.data[f'{col_name}_high_vol'] = (self.data[f'{col_name}_pct'] > 2.0).astype(int)
            self.data[f'{col_name}_trend'] = self.data[col_name].diff()
            
            added_columns.extend([col_name, f'{col_name}_pct', 
                                f'{col_name}_zscore', f'{col_name}_high_vol',
                                f'{col_name}_trend'])
        
        print(f'New columns added: {", ".join(added_columns)}')
        print("="*50)
        
        return self.data
    
    def get_all_volatility_indicators(self,
                                     atr_periods: List[int] = [14, 21, 28],
                                     bb_periods: List[int] = [20, 50],
                                     keltner_ema_period: int = 20,
                                     keltner_atr_period: int = 10,
                                     keltner_multiplier: float = 2.0,
                                     std_periods: List[int] = [20, 50, 100]):
        
        """
        Adds all volatility indicators
        
        Parameters:
        atr_periods (List[int]): List of periods for ATR
        bb_periods (List[int]): List of periods for Bollinger Bands
        keltner_ema_period (int): EMA period for Keltner Channels
        keltner_atr_period (int): ATR period for Keltner Channels
        keltner_multiplier (float): Multiplier for Keltner Channels
        std_periods (List[int]): List of periods for Standard Deviation
        
        """
        
        self.add_atr(atr_periods)
        self.add_bollinger_bands(bb_periods)
        self.add_keltner_channels(keltner_ema_period, keltner_atr_period, keltner_multiplier)
        self.add_standard_deviation(std_periods)
        # self.add_volatility_confirmation()
        # self.get_volatility_score()
        
        return self.data