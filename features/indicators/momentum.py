import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexMomentumIndicators:
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
        Class for Momentum Indicators
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        open_col (str): Column name for open price
        high_col (str): Column name for high price
        low_col (str): Column name for low price
        close_col (str): Column name for close price
        volume_col (str): Column name for volume
        
        """
        
        print("="*50)
        print("MOMENTUM INDICATORS")
        print("="*50)
        print(" Available Fuctions \n1 add_rsi \n2 add_stochastic \n3 add_williams_r \n4 add_cci \n5 add_momentum \n6 generate_all_momentum_indicators")
        print("="*50) 
        
        self.data = data.copy()
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        
        self.momentum_data = pd.DataFrame(
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
    
    def add_rsi(
        self, 
        periods: List[int] = [14, 21, 28],
    ):
        
        """
        Relative Strength Index
        
        Parameters:
        periods (List[int]): List of periods for RSI
        
        """
         
        for period in periods:
            col_name = f'rsi_{period}'
            
            # RSI
            self.momentum_data[col_name] = talib.RSI(
                self.data[self.close_col], 
                timeperiod=period
            )
            
            #RSI slope
            self.momentum_data[f'{col_name}_slope'] = self.momentum_data[col_name].diff()
        
        return self.momentum_data
    
    def add_stochastic(
        self,
        fk_sk_sd_periods: List[int] = [14, 3, 3],
    ):
        
        """
        Stochastic Oscillator
        
        Parameters:
        fk_sk_sd_periods (List[int]): List nested or not, [fastk_period, slowk_period, slowd_period] for Stochastic
        
        """
        
        is_nested = self._is_nested_list(fk_sk_sd_periods)
        
        if is_nested:
            for lst in fk_sk_sd_periods:
                fast_k = lst[0]
                slow_k = lst[1]
                slow_d = lst[2]
                
                slowk, slowd = talib.STOCH(
                    self.data[self.high_col],
                    self.data[self.low_col],
                    self.data[self.close_col],
                    fastk_period = fast_k,
                    slowk_period = slow_k,
                    slowk_matype = 0,
                    slowd_period = slow_d,
                    slowd_matype = 0
                )
                
                col_namek = f'stoch_slowk_{slow_k}'
                col_named = f'stoch_slowd_{slow_d}'
                
                # Stochastic
                self.momentum_data[col_namek] = slowk
                self.momentum_data[col_named] = slowd
                
                # Stochastic slope
                self.momentum_data[f'{col_namek}_slope'] = self.momentum_data[col_namek].diff()
                self.momentum_data[f'{col_named}_slope'] = self.momentum_data[col_named].diff()
        else:
            fast_k = fk_sk_sd_periods[0]
            slow_k = fk_sk_sd_periods[1]
            slow_d = fk_sk_sd_periods[2]
            
            slowk, slowd = talib.STOCH(
                self.data[self.high_col],
                self.data[self.low_col],
                self.data[self.close_col],
                fastk_period = fast_k,
                slowk_period = slow_k,
                slowk_matype = 0,
                slowd_period = slow_d,
                slowd_matype = 0
            )
            
            col_namek = f'stoch_slowk_{slow_k}'
            col_named = f'stoch_slowd_{slow_d}'
            
            # Stochastic
            self.momentum_data[col_namek] = slowk
            self.momentum_data[col_named] = slowd
            
            # Stochastic slope
            self.momentum_data[f'{col_namek}_slope'] = self.momentum_data[col_namek].diff()
            self.momentum_data[f'{col_named}_slope'] = self.momentum_data[col_named].diff()
                
        return self.momentum_data
    
    def add_williams_r(
        self,
        periods: List[int] = [14, 21, 28],
    ):
        
        """
        Williams %R
        
        Parameters:
        periods (List[int]): List of periods for Williams %R
        
        """
        
        for period in periods:
            willr = talib.WILLR(
                self.data[self.high_col],
                self.data[self.low_col],
                self.data[self.close_col],
                timeperiod = period
            )
            
            col_name = f'williams_r_{period}'
            
            # Williams %R
            self.momentum_data[col_name] = willr
            
            #Williams %R slope
            self.momentum_data[f'{col_name}_slope'] = willr.diff()
                    
        return self.momentum_data
    
    def add_cci(
        self,
        periods: List[int] = [14, 21, 28],
    ):
        
        """
        Commodity Channel Index
        
        Parameters:
        periods (List[int]): List of periods for CCI
        
        """
        
        for period in periods:
            cci = talib.CCI(
                self.data[self.high_col],
                self.data[self.low_col],
                self.data[self.close_col],
                timeperiod = period
            )
            
            col_name = f'cci_{period}'
            
            # CCI
            self.momentum_data[col_name] = cci
            
            # CCI slope
            self.momentum_data[f'{col_name}_slope'] = cci.diff()
                
        return self.momentum_data
    
    def add_momentum(
        self, 
        periods: List[int] = [10, 14, 20],
    ):
        
        """
        Momentum Indicator
        
        Parameters:
        periods (List[int]): List of periods for Momentum
        
        """
        
        for period in periods:
            
            col_name = f'momentum_{period}'
            
            # Momentum
            self.momentum_data[col_name] = talib.MOM(
                self.data[self.close_col], 
                timeperiod=period
            )
            
            # Momentum slope
            self.momentum_data[f'{col_name}_slope'] = self.momentum_data[col_name].diff()
        
        return self.momentum_data
    
    def generate_all_momentum_indicators(
        self,
        rsi_periods: List[int] = [14, 21, 28],
        stochastic_periods: List[int] = [14, 3, 3],
        williams_periods: List[int] = [14, 21, 28],
        cci_periods: List[int] = [14, 21, 28],
        momentum_periods: List[int] = [10, 14, 20],
    ):
        
        """
        Adds all momentum indicators
        
        Parameters:
        
        rsi_periods (List[int]): List of periods for RSI
        stochastic_periods (List[int]): List nested or not, [fastk_period, slowk_period, slowd_period]
        williams_periods (List[int]): List of periods for Williams %R
        cci_periods (List[int]): List of periods for CCI
        momentum_periods (List[int]): List of periods for Momentum
        
        """
        
        self.add_rsi(periods = rsi_periods)
        self.add_stochastic(fk_sk_sd_periods = stochastic_periods)
        self.add_williams_r(periods = williams_periods)
        self.add_cci(periods = cci_periods)
        self.add_momentum(periods = momentum_periods)
        
        count_removed_rows = self.data.shape[0] - self.momentum_data.shape[0]
        
        print('='*50)
        print('Data Info')
        print(self.momentum_data.info())
        print('='*50)
        print(f'Shape of data {self.momentum_data.shape}')
        print('='*50)
        print(f'{count_removed_rows} rows removed')
        print('='*50)
        
        return self.momentum_data