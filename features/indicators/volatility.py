import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexVolatilityIndicators:
    def __init__(
        self, 
        data: pd.DataFrame,
        open_col: str = 'open',
        high_col: str = 'high', 
        low_col: str = 'low', 
        close_col: str = 'close',
        volume_col: str = 'volume',
    ):
        
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
        print(" Available Fuctions \n1 add_atr \n2 add_bollinger_bands \n3 add_keltner_channels \n4 add_standard_deviation  \n5 generate_all_volatility_indicators")
        print("="*50)
        
        self.data = data.copy()
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        
        self.volatility_data = pd.DataFrame(
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
            self.high_col,
            self.low_col,
            self.open_col,
            self.volume_col
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
    
    def add_atr(
        self, 
        periods: List[int] = [10, 14, 21, 28],
    ):
        
        """
        Average True Range
        
        Parameters:
        periods (List[int]): List of periods for ATR
        
        """

        for period in periods:
            col_name = f'atr_{period}'
            
            self.volatility_data[col_name] = talib.ATR(
                self.data[self.high_col],
                self.data[self.low_col],
                self.data[self.close_col],
                timeperiod=period
            )

        return self.volatility_data
    
    def add_bollinger_bands(
        self,
        period_nbdevup_nbdevdn: List = [20, 2.0, 2.0],
    ):

        """
        Bollinger Bands
        
        Parameters:
        period_nbdevup_nbdevdn (List[int, float, float]): List nested or not, [period, nbdevup, nbdevdn]
        
        """
        
        is_nested = self._is_nested_list(period_nbdevup_nbdevdn)
        
        if is_nested:
            for lst in period_nbdevup_nbdevdn:
                timeperiod = lst[0]
                nbdevup = lst[1]
                nbdevdn = lst[2]
                
                upper, middle, lower = talib.BBANDS(
                    self.data[self.close_col],
                    timeperiod=timeperiod,
                    nbdevup=nbdevup,
                    nbdevdn=nbdevdn
                )
                
                col_name = f'bb_{timeperiod}'
                
                self.volatility_data[f'{col_name}_upper'] = upper
                self.volatility_data[f'{col_name}_middle'] = middle
                self.volatility_data[f'{col_name}_lower'] = lower
        else:   
            timeperiod = period_nbdevup_nbdevdn[0]
            nbdevup = period_nbdevup_nbdevdn[1]
            nbdevdn = period_nbdevup_nbdevdn[2]
            
            upper, middle, lower = talib.BBANDS(
                self.data[self.close_col],
                timeperiod=timeperiod,
                nbdevup=nbdevup,
                nbdevdn=nbdevdn 
            )   
            
            col_name = f'bb_{timeperiod}'
            
            self.volatility_data[f'{col_name}_upper'] = upper
            self.volatility_data[f'{col_name}_middle'] = middle
            self.volatility_data[f'{col_name}_lower'] = lower
        
        return self.volatility_data
    
    def add_keltner_channels(
        self,
        ema_atr_multiplier: List = [20, 10, 2.0],
    ):

        """
        keltner Channels
        
        Parameters:
        ema_atr_multiplier (List): List nested or not, [EMA period, ATR period, ATR multiplier]
        
        """
        
        is_nested = self._is_nested_list(ema_atr_multiplier)
        
        if is_nested:
            for lst in ema_atr_multiplier:
                ema_col = f'ema_{lst[0]}'
                atr_col = f'atr_{lst[1]}'
                multiplier = lst[2]
                
                if ema_col not in self.data.columns:            
                    self.data[ema_col] = talib.EMA(
                        self.data[self.close_col], 
                        timeperiod=lst[0]
                    )

                if atr_col not in self.volatility_data.columns:
                    self.add_atr(periods=[lst[1]])
                
                col_name = f'keltner_{ema_col}_{atr_col}'

                self.volatility_data[f'{col_name}_middle'] = self.data[ema_col]
                self.volatility_data[f'{col_name}_upper'] = self.data[ema_col] + (self.volatility_data[atr_col] * multiplier) 
                self.volatility_data[f'{col_name}_lower'] = self.data[ema_col] - (self.volatility_data[atr_col] * multiplier)
        else:
            
            ema_col = f'ema_{ema_atr_multiplier[0]}'
            atr_col = f'atr_{ema_atr_multiplier[1]}'
            multiplier = ema_atr_multiplier[2]
            if ema_col not in self.data.columns:            
                self.data[ema_col] = talib.EMA(
                    self.data[self.close_col], 
                    timeperiod=ema_atr_multiplier[0]
                )
                    
            if atr_col not in self.volatility_data.columns:
                    self.add_atr(periods=[ema_atr_multiplier[1]])
            
            col_name = f'keltner_{ema_col}_{atr_col}'
            
            self.volatility_data[f'{col_name}_middle'] = self.data[ema_col]
            self.volatility_data[f'{col_name}_upper'] = self.data[ema_col] + (self.volatility_data[atr_col] * multiplier) 
            self.volatility_data[f'{col_name}_lower'] = self.data[ema_col] - (self.volatility_data[atr_col] * multiplier)
                
        return self.volatility_data
    
    def add_standard_deviation(
        self, 
        periods: List[int] = [20, 50, 100],
    ):
        
        """
        Standard Deviation
        
        Parameters:
        periods (List[int]): List of periods for Standard Deviation
        
        """
        
        for period in periods:
            col_name = f'vol_std_dev_{period}'
            
            self.volatility_data[col_name] = talib.STDDEV(
                self.data[self.close_col],
                timeperiod=period,
                nbdev=1
            )
        
        return self.volatility_data
    
    def generate_all_volatility_indicators(
        self,
        atr_periods: List[int] = [14, 21, 28],
        bb_period_nbdevup_nbdevdn: List = [20, 2.0, 2.0],
        kethlner_ema_atr_multiplier: List = [20, 10, 2.0],
        std_periods: List[int] = [20, 50, 100],
    ):
        
        """
        Adds all volatility indicators
        
        Parameters:
        atr_periods (List[int]): List of periods for ATR
        bb_period_nbdevup_nbdevdn (List[int, float, float]): List nested or not, [period, nbdevup, nbdevdn]
        kethlner_ema_atr_multiplier (List[int, int, float]): List nested or not, [EMA period, ATR period, ATR multiplier]
        std_periods (List[int]): List of periods for Standard Deviation
        
        """
        
        self.add_atr(periods = atr_periods)
        self.add_bollinger_bands(period_nbdevup_nbdevdn = bb_period_nbdevup_nbdevdn )
        self.add_keltner_channels(ema_atr_multiplier = kethlner_ema_atr_multiplier)
        self.add_standard_deviation(periods = std_periods)
        
        count_removed_rows = self.data.shape[0] - self.volatility_data.shape[0]
        
        print('='*50)
        print('Data Info')
        print(self.volatility_data.info())
        print('='*50)
        print(f'Shape of data {self.volatility_data.shape}')
        print('='*50)
        print(f'{count_removed_rows} rows removed')
        print('='*50)
        
        return self.volatility_data