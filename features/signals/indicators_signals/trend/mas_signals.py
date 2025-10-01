import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

from itertools import combinations

import warnings
warnings.filterwarnings('ignore')

class ForexMASignals:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
        sma_parameters: List = None,
        ema_parameters: List = None,
    ):
        
        """
        Class for EMA/SMA signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
        print("="*50)
        print("EMA/SMA SIGNAL GENERATION")
        print("="*50)   
        print(" Available Fuctions: \n1 golden_death_cross \n2 ema_crossover \n3 trend_hierarchy \n4 ma_bounce_signals \n5 ma_slope_signals \n6 price_extension_signals \n7 generate_all_signals")
        print("="*50)
        
        self.data = data.copy()
        self.close_col = close_col
        
        self.signals = pd.DataFrame(
            {self.close_col: self.data[self.close_col]}, 
            index=self.data.index
        )
        
        self.ema_names = []
        self.ema_slope_names = []
        
        self.sma_names = []
        self.sma_slope_names = []
        
        self.ema_parameters = [10, 20, 50, 100, 200] if ema_parameters is None else ema_parameters
        self.sma_parameters = [10, 20, 50, 100, 200] if sma_parameters is None else sma_parameters
        
        self._validate_columns()
        self._extract_column_names(parameters=self.ema_parameters, ma_type='ema')
        self._extract_column_names(parameters=self.sma_parameters, ma_type='sma')
                
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
        self,
        parameters: List = None,
        ma_type: str = None
    ):
        
        """
        Extract MA column names based on parameters
        
        """
        names = []
        slope_names = []
        
        is_nested = self._is_nested_list(parameters)
        
        if is_nested:
            all_periods = []
            for sublist in parameters:
                all_periods.extend(sublist)
            periods = all_periods
        else:
            periods = parameters
        
        for period in periods:
            ma_name = f'{ma_type}_{period}'
            slope_name = f'{ma_name}_slope'
            self._validate_columns([ma_name, slope_name])
            names.append(ma_name)
            slope_names.append(slope_name)
            
        if ma_type == 'ema':
            self.ema_names.extend(names)
            self.ema_slope_names.extend(slope_names)
        elif ma_type == 'sma':
            self.sma_names.extend(names)
            self.sma_slope_names.extend(slope_names)
            
    def golden_death_cross(
        self, 

    ):
        
        """
        Golden Cross (2) when fast SMA crosses above slow SMA
        Death Cross (1) when fast SMA crosses below slow SMA  
        No signal (0) for no crossover
    
        Parameters:
        fast_col (str): Column name for fast SMA
        slow_col (str): Column name for slow SMA
        
        """
        
        for name1, name2 in combinations(self.sma_names, 2):
            sorted_ma = sorted([name1, name2], key = lambda x: int(x.split('_')[1]))
            fast, slow = sorted_ma
            
            golden_condition = (
                (self.data[fast] > self.data[slow]) & 
                (self.data[fast].shift(1) <= self.data[slow].shift(1))
            )

            death_condition = (
                (self.data[fast] < self.data[slow]) & 
                (self.data[fast].shift(1) >= self.data[slow].shift(1))
            )
            
            self.signals[f'{fast}_{slow}_golden_death_cross'] = np.select(
                [golden_condition, death_condition],
                [2, 1],
                default = 0
            )
                
        return self.signals
        
        # columns = f'trend_sma_{fast_col}', f'trend_sma_{slow_col}'
        # self._validate_columns(columns = columns)
        
        # for name1 in self.sma_names:
        #     for name2 in self.sma_names:
        #         period1 = int(name1.split('_')[1])
        #         period2 = int(name2.split('_')[1])
        #         if name1 != name2 and period1 < period2 :
        #             # Golden Cross: fast crosses above slow
        #             golden_condition = (
        #                 (self.data[name1] > self.data[name2]) & 
        #                 (self.data[name1].shift(1) <= self.data[name2].shift(1))
        #             )
                    
        #             # Death Cross: fast crosses below slow  
        #             death_condition = (
        #                 (self.data[name1] < self.data[name2]) & 
        #                 (self.data[name1].shift(1) >= self.data[name2].shift(1))
        #             )
                    
        #             # Create signal for entire series: 1=Golden, 0=Death, NaN=No signal
        #             self.signals[f'{name1}_{name2}_golden_death_cross'] = np.select(
        #                 [golden_condition, death_condition],
        #                 [2, 1],
        #                 default = 0
                
        # return self.signals
    
    def ema_crossover(
        self,
    ):
        
        """
        EMA Crossover signals (Bullish: EMAfast > EMAslow, Bearish: EMAfast < EMAslow)
        
        """
        
        for name1, name2 in combinations(self.ema_names, 2):
            sorted_double = sorted([name1, name2], key=lambda x: int(x.split('_')[1]))
            fast, slow = sorted_double
            self._validate_columns(columns = [fast, slow])
            
            bullish_condition = (
                (self.data[fast] > self.data[slow]) & 
                (self.data[fast].shift(1) <= self.data[slow].shift(1))
            )
            
            bearish_condition = (
                (self.data[fast] < self.data[slow]) & 
                (self.data[fast].shift(1) >= self.data[slow].shift(1))
            )
            
            # Create signal
            self.signals[f'{fast}_{slow}_crossover'] = np.select(
                [bullish_condition, bearish_condition],
                [2, 1],
                default=0
            )
            
        return self.signals
            
    def trend_hierarchy(
        self,
    ):
        
        """
        Checks if MAs are in perfect bullish/bearish alignment
        
        """
        
        for ma_type in ['ema', 'sma']:
            names = self.ema_names if ma_type == 'ema' else self.sma_names
            for name1, name2, name3 in combinations(names, 3):
                sort_names = sorted([name1, name2, name3], key=lambda x: int(x.split('_')[1]))
                fast, mid, slow = sort_names
                self._validate_columns(columns = [fast, mid, slow])
                
                bullish_condition = (
                    (self.data[fast] > self.data[mid]) & 
                    (self.data[mid] > self.data[slow])
                )

                bearish_condition = (
                    (self.data[fast] < self.data[mid]) & 
                    (self.data[mid] < self.data[slow])
                )
                
                self.signals[f'{fast}_{mid}_{slow}_hierarchy'] = np.select(
                    [bullish_condition, bearish_condition],
                    [2, 1],
                    default = 0
                )
        
        return self.signals
        
        # columns = []
        # for period in periods:
        #     columns.append(f'trend_ema_{period}')
            
        # self._validate_columns(columns = columns)
        
        # bullish_condition = True
        # bearish_condition = True
        
        # for i in range(len(columns)-1):
        #     if columns[i] > columns[i+1]:
        #         bullish = False
        #     elif columns[i] < columns[i+1]:
        #         bearish = False

        # self.signals['bearish_bullish_hierarchy'] = np.select(
        #     [bullish_condition, bearish_condition],
        #     [2, 1],
        #     default = 0
        # )
    
        # return self.signals
    
    def ma_bounce_signals(
        self, 
    ):
        
        """
        Signals when price bounces off moving average support/resistance
        
        """
        
        for ma_type in ['ema', 'sma']:
            names = self.ema_names if ma_type == 'ema' else self.sma_names
            for name in names:
                self._validate_columns(columns = [name])
                threshold = self.data[name] * 0.001
                price_touch_ma = abs(self.data[self.close_col] - self.data[name]) <= threshold

                bearish_bounce = (
                    (self.data[self.close_col].shift(1) > self.data[name].shift(1)) &
                    price_touch_ma &
                    (self.data[self.close_col] < self.data[name])
                )
                
                bullish_bounce = (
                    (self.data[self.close_col].shift(1) < self.data[name].shift(1)) &
                    price_touch_ma &
                    (self.data[self.close_col] > self.data[name])
                )
                
                self.signals[f'{name}_bounce'] = np.select(
                    [bearish_bounce, bullish_bounce],
                    [1, 2],
                    default = 0
                )
                
                return self.signals
        
        # column = f'trend_{ma_type}_{period}'
        # self._validate_columns(columns = [column])
        
        # touch_threshold = self.data[column] * 0.001
        # price_touches_ma = abs(self.data[self.close_col] - self.data[column]) <= touch_threshold
        
        # bearish_bounce = (
        #     (self.data[self.close_col].shift(1) > self.data[column].shift(1)) &
        #     price_touches_ma &
        #     (self.data[self.close_col] < self.data[column])
        # )
        
        # bullish_bounce = (
        #     (self.data[self.close_col].shift(1) < self.data[column].shift(1)) &
        #     price_touches_ma &
        #     (self.data[self.close_col] > self.data[column])
        # )
        
        # self.signals['bounce_bearish_bullish'] = np.select(
        #     [bearish_bounce, bullish_bounce],
        #     [1, 2],
        #     default = 0
        # )
        
        # return self.signals
        
    def ma_slope_signals(
        self,  
        # lookback: int = 3
    ):
        
        """
        Signals based on moving average slope and acceleration
        
        """
        
        for ma_type in ['ema', 'sma']:
            names = self.ema_names if ma_type == 'ema' else self.sma_names
            for name in names:
                self._validate_columns(columns = [name, f'{name}_slope'])
                
                positive_slope = self.data[f'{name}_slope'] > 0
                negative_slope = self.data[f'{name}_slope'] < 0
                
                self.signals[f'{name}_slope_direction'] = np.select(
                    [positive_slope, negative_slope],
                    [2, 1],
                    default = 0
                )
                
                slope_increasing = self.data[f'{name}_slope'] > self.data[f'{name}_slope'].shift(1)
                slope_decreasing = self.data[f'{name}_slope'] < self.data[f'{name}_slope'].shift(1)

                self.signals[f'{name}_slope_acceleration'] = np.select(
                    [slope_increasing, slope_decreasing],
                    [2, 1],  
                    default = 0  
                )
                
                strong_uptrend = (self.data[self.close_col] > self.data[name]) & (self.data[f'{name}_slope'] > 0)
                strong_downtrend = (self.data[self.close_col] < self.data[name]) & (self.data[f'{name}_slope'] < 0)

                self.signals[f'{name}_trend_strong'] = np.select(
                    [strong_uptrend, strong_downtrend],
                    [2, 1],  
                    default = 0  
                )
                
        return self.signals 
        
        # columns = [f"trend_{ma_type}_{period}", f"trend_{ma_type}_{period}_slope"]
        # self._validate_columns(columns = columns)
    
        # # Positive/Negative slope 
        # positive_slope = self.data[columns[1]] > 0
        # negative_slope = self.data[columns[1]] < 0
        
        # self.signals['slope_direction'] = np.select(
        #     [positive_slope, negative_slope],
        #     [2, 1],
        #     default = 0
        # )
        
        # # Slope acceleration
        # slope_increasing = self.data[columns[1]] > self.data[columns[1]].shift(1)
        # slope_decreasing = self.data[columns[1]] < self.data[columns[1]].shift(1)

        # self.signals['slope_acceleration'] = np.select(
        #     [slope_increasing, slope_decreasing],
        #     [2, 1],  
        #     default = 0  
        # )
        
        # # Strong trend signals
          
        # strong_uptrend = (self.data[self.close_col] > self.data[columns[0]]) & (self.data[columns[1]] > 0)
        # strong_downtrend = (self.data[self.close_col] < self.data[columns[0]]) & (self.data[columns[1]] < 0)

        # self.signals['trend_strong'] = np.select(
        #     [strong_uptrend, strong_downtrend],
        #     [2, 1],  
        #     default = 0  
        # )
        
    def price_extension_signals(
        self, 
        deviation: float = 0.02
    ):
        
        """
        Signals when price is overextended from moving average
        
        """
        
        for ma_type in ['ema', 'sma']:
            names = self.ema_names if ma_type == 'ema' else self.sma_names
            for name in names:
                self._validate_columns(columns = [name])
                
                deviation_pct = abs(self.data[self.close_col] - self.data[name]) / self.data[name]

                overbought = (
                    (self.data[self.close_col] > self.data[name]) & 
                    (deviation_pct > deviation)
                )
                
                oversold = (
                    (self.data[self.close_col] < self.data[name]) & 
                    (deviation_pct > deviation)
                )
                
                self.signals[f'{name}_overbought_oversold'] = np.select(
                    [overbought, oversold],
                    [2, 1],
                    default = 0
                )
                
        return self.signals
        
        # column = f'trend_{ma_type}_{period}'
        # self._validate_columns(columns = [column])
        
        # # Calculate percentage deviation from MA
        # deviation_pct = abs(self.data[self.close_col] - self.data[column]) / self.data[column]
        
        # # Overbought: Price significantly above MA
        # overbought = ( 
        #     (self.data[self.close_col] > self.data[column]) & 
        #     (deviation_pct > deviation)
        # )
        
        # # Oversold: Price significantly below MA    
        # oversold = (
        #     (self.data[self.close_col] < self.data[column]) & 
        #     (deviation_pct > deviation)
        # )
        
        # self.signals["overbought_oversold"] = np.select(
        #     [overbought, oversold],
        #     [2, 1],
        #     default = 0
        # )
        
        # return self.signals
        
    def generate_all_signals(
        self,
        price_extesion_deviation: float = 0.02,   
    ): 
        
        """
        Generate all available SMA/EMA signals
        
        """
        
        self.golden_death_cross()
        self.ema_crossover()
        self.trend_hierarchy()
        self.ma_bounce_signals()
        self.ma_bounce_signals()
        self.ma_slope_signals()
        self.ma_slope_signals()
        self.price_extension_signals(deviation = price_extesion_deviation)

        count_removed_rows = self.data.shape[0] - self.signals.shape[0]
        
        print('='*50)
        print('Data Info')
        print(self.signals.info())
        print('='*50)   
        print(f'Shape of data {self.signals.shape}')
        print('='*50)
        print(f'{count_removed_rows} rows removed')
        print('='*50)
        
        return self.signals


