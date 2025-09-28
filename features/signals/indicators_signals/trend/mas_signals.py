import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexMASignals:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
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
        
        self._validate_columns()
        
    def _validate_columns(
        self, 
        columns: List[str] =  None,
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
            
    def golden_death_cross(
        self, 
        fast_col: int = 50, 
        slow_col: int = 200
    ):
        
        """
        Golden Cross (2) when fast SMA crosses above slow SMA
        Death Cross (1) when fast SMA crosses below slow SMA  
        No signal (0) for no crossover
    
        Parameters:
        fast_col (str): Column name for fast SMA
        slow_col (str): Column name for slow SMA
        
        """
        
        columns = f'trend_sma_{fast_col}', f'trend_sma_{slow_col}'
        self._validate_columns(columns = columns)
 
            # Golden Cross: fast crosses above slow
        golden_condition = (
            (self.data[columns[0]] > self.data[columns[1]]) & 
            (self.data[columns[0]].shift(1) <= self.data[columns[1]].shift(1))
        )
        
        # Death Cross: fast crosses below slow  
        death_condition = (
            (self.data[columns[0]] < self.data[columns[1]]) & 
            (self.data[columns[0]].shift(1) >= self.data[columns[1]].shift(1))
        )
        
        # Create signal for entire series: 1=Golden, 0=Death, NaN=No signal
        self.signals['golden_death_cross'] = np.select(
            [golden_condition, death_condition],
            [2, 1],
            default = 0
        )
    
   
        return self.signals
    
    def ema_crossover(
        self, 
        fast_col=20, 
        slow_col=50
    ):
        
        """
        EMA Crossover signals (Bullish: EMA20 > EMA50, Bearish: EMA20 < EMA50)
        
        """
        
        columns = [f'trend_ema_{fast_col}', f'trend_ema_{slow_col}']
        self._validate_columns(columns = columns)
        
        bullish_condition = (
            (self.data[columns[0]] > self.data[columns[1]]) & 
            (self.data[columns[0]].shift(1) <= self.data[columns[1]].shift(1))
        )
        
        bearish_condition = (
            (self.data[columns[0]] < self.data[columns[1]]) & 
            (self.data[columns[0]].shift(1) >= self.data[columns[1]].shift(1))
        )
        
        self.signals['ema_crossover'] = np.select(
            [bullish_condition, bearish_condition],
            [2, 1],
            default = 0
        )
        
    def trend_hierarchy(
        self,
        periods: List[int] = [50, 100, 200]
    ):
        
        """
        Checks if EMAs are in perfect bullish/bearish alignment
        
        Parameters:
        periods (List[int]): List of periods for EMA
        
        """
        
        columns = []
        for period in periods:
            columns.append(f'trend_ema_{period}')
            
        self._validate_columns(columns = columns)
        
        bullish_condition = True
        bearish_condition = True
        
        for i in range(len(columns)-1):
            if columns[i] > columns[i+1]:
                bullish = False
            elif columns[i] < columns[i+1]:
                bearish = False

        self.signals['bearish_bullish_hierarchy'] = np.select(
            [bullish_condition, bearish_condition],
            [2, 1],
            default = 0
        )
    
        return self.signals
    
    def ma_bounce_signals(
        self, 
        period=20, 
        ma_type='ema'
    ):
        
        """
        Signals when price bounces off moving average support/resistance
        
        Parameters:
        ma_period (int): Period for moving average
        ma_type (str): Type of moving average
        
        """
        
        column = f'trend_{ma_type}_{period}'
        self._validate_columns(columns = [column])
        
        touch_threshold = self.data[column] * 0.001
        price_touches_ma = abs(self.data[self.close_col] - self.data[column]) <= touch_threshold
        
        bearish_bounce = (
            (self.data[self.close_col].shift(1) > self.data[column].shift(1)) &
            price_touches_ma &
            (self.data[self.close_col] < self.data[column])
        )
        
        bullish_bounce = (
            (self.data[self.close_col].shift(1) < self.data[column].shift(1)) &
            price_touches_ma &
            (self.data[self.close_col] > self.data[column])
        )
        
        self.signals['bounce_bearish_bullish'] = np.select(
            [bearish_bounce, bullish_bounce],
            [1, 2],
            default = 0
        )
        
        return self.signals
        
    def ma_slope_signals(
        self, 
        period: int = 20, 
        ma_type: str = 'ema', 
        lookback: int = 3
    ):
        
        """
        Signals based on moving average slope and acceleration
        
        Parameters:
        period (int): Period for moving average
        ma_type (str): Type of moving average
        lookback (int): Number of periods to look back for slope
        
        """
        
        columns = [f"trend_{ma_type}_{period}", f"trend_{ma_type}_{period}_slope"]
        self._validate_columns(columns = columns)
    
        # Positive/Negative slope 
        positive_slope = self.data[columns[1]] > 0
        negative_slope = self.data[columns[1]] < 0
        
        self.signals['slope_direction'] = np.select(
            [positive_slope, negative_slope],
            [2, 1],
            default = 0
        )
        
        # Slope acceleration
        slope_increasing = self.data[columns[1]] > self.data[columns[1]].shift(1)
        slope_decreasing = self.data[columns[1]] < self.data[columns[1]].shift(1)

        self.signals['slope_acceleration'] = np.select(
            [slope_increasing, slope_decreasing],
            [2, 1],  
            default = 0  
        )
        
        # Strong trend signals
          
        strong_uptrend = (self.data[self.close_col] > self.data[columns[0]]) & (self.data[columns[1]] > 0)
        strong_downtrend = (self.data[self.close_col] < self.data[columns[0]]) & (self.data[columns[1]] < 0)

        self.signals['trend_strong'] = np.select(
            [strong_uptrend, strong_downtrend],
            [2, 1],  
            default = 0  
        )
        
    def price_extension_signals(
        self, 
        period: int = 20, 
        ma_type: str = 'ema', 
        deviation: float = 0.02
    ):
        
        """
        Signals when price is overextended from moving average
        
        Parameters:
        period (int): Period for moving average
        ma_type (str): Type of moving average
        deviation (float): Deviation threshold for extension
        
        """
        
        column = f'trend_{ma_type}_{period}'
        self._validate_columns(columns = [column])
        
        # Calculate percentage deviation from MA
        deviation_pct = abs(self.data[self.close_col] - self.data[column]) / self.data[column]
        
        # Overbought: Price significantly above MA
        overbought = ( 
            (self.data[self.close_col] > self.data[column]) & 
            (deviation_pct > deviation)
        )
        
        # Oversold: Price significantly below MA    
        oversold = (
            (self.data[self.close_col] < self.data[column]) & 
            (deviation_pct > deviation)
        )
        
        self.signals["overbought_oversold"] = np.select(
            [overbought, oversold],
            [2, 1],
            default = 0
        )
        
        return self.signals
        
    def generate_all_signals(self): #CHANGE PARAMETERS TOMMOEOW
        
        """
        Generate all available SMA/EMA signals
        
        """
        
        self.golden_death_cross()
        self.ema_crossover()
        self.trend_hierarchy()
        self.ma_bounce_signals(20, 'ema')
        self.ma_bounce_signals(50, 'sma')
        self.ma_slope_signals(20, 'ema')
        self.ma_slope_signals(50, 'sma')
        self.price_extension_signals(20, 'ema', 0.02)
        print(self.signals.tail(10), "\n", self.signals.shape)
        return self.signals


