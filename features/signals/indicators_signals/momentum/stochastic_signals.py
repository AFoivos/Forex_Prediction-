import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexStochasticSignals:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
        slowk_col: str = 'momen_stoch_slowk',
        slowd_col: str = 'momen_stoch_slowd',
        slopek_col: str = 'momen_stoch_slowk_slope',
        sloped_col: str = 'momen_stoch_slowd_slope',
    ):
        
        """
        Class for Stochastic signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
        print("="*50)
        print("STOCHASTIC SIGNAL GENERATION")
        print("="*50)
        print("Available functions: \n1 stochastic_overbought_oversold_signals \n2 stochastic_crossover_signals \n3 stochastic_divergence_signals \n4 stochastic_k_momentum_signals \n5 stochastic_d_momentum_signals \n6 stochastic_reversal_signals \n7 generate_all_stochastic_signals")
        print("="*50)
        
        self.close_col = close_col
        self.data = data.copy()
        
        self.slowκ = slowk_col
        self.slowd = slowd_col
        self.slopek = slopek_col
        self.sloped = sloped_col
        
        self._validate_columns()
        
        self.signals = pd.DataFrame(
            {'close': self.data['close']},
            index=self.data.index
        )
    
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
            self.slowκ,
            self.slowd,
            self.slopek,
            self.sloped
        ]
        
        if columns is not None:
            required_cols.extend(columns)
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    def stochastic_overbought_oversold_signals(
        self, 
        overbought: int = 80, 
        oversold: int = 20
    ):
        """
        Stochastic Overbought/Oversold Signals
        2 = Overbought (Stochastic > 80)
        1 = Oversold (Stochastic < 20)
        0 = Normal (20 <= Stochastic <= 80)
        
        Parameters:
        overbought (int): Overbought threshold
        oversold (int): Oversold threshold
        
        """
        
        overbought_condition = self.data[self.slowκ] > overbought
        oversold_condition = self.data[self.slowκ] < oversold
        
        self.signals['stoch_overbought_oversold'] = np.select(
            [overbought_condition, oversold_condition],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def stochastic_crossover_signals(self):
        
        """
        Stochastic %K/%D Crossover Signals
        2 = Bullish (%K crosses above %D)
        1 = Bearish (%K crosses below %D)
        0 = No crossover
    
        """
        
        bullish_cross = (
            (self.data[self.slowκ] > self.data[self.slowd]) & 
            (self.data[self.slowκ].shift(1) <= self.data[self.slowd].shift(1))
        )
        
        bearish_cross = (
            (self.data[self.slowκ] < self.data[self.slowd]) & 
            (self.data[self.slowκ].shift(1) >= self.data[self.slowd].shift(1))
        )
        
        self.signals['stoch_crossover'] = np.select(
            [bullish_cross, bearish_cross],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def stochastic_divergence_signals(
        self,
        lookback: int = 10
    ):
        
        """
        Stochastic Divergence Signals
        2 = Bullish Divergence (Price Lower Low, Stochastic Higher Low)
        1 = Bearish Divergence (Price Higher High, Stochastic Lower High)
        0 = No Divergence
        
        Parameters:
        lookback (int): Lookback period for Stochastic
        
        """
        
        # Bullish Divergence: Price Lower Low, Stochastic Higher Low
        price_lower_low = (
            (self.data[self.close_col] < self.data[self.close_col].shift(lookback)) &
            (self.data[self.close_col].shift(1) < self.data[self.close_col].shift(lookback + 1))
        )
        stoch_higher_low = (
            (self.data[self.slowκ] > self.data[self.slowκ].shift(lookback)) &
            (self.data[self.slowκ].shift(1) > self.data[self.slowκ].shift(lookback + 1))
        )
        bullish_divergence = price_lower_low & stoch_higher_low
        
        # Bearish Divergence: Price Higher High, Stochastic Lower High
        price_higher_high = (
            (self.data[self.close_col] > self.data[self.close_col].shift(lookback)) &
            (self.data[self.close_col].shift(1) > self.data[self.close_col].shift(lookback + 1))
        )
        stoch_lower_high = (
            (self.data[self.slowκ] < self.data[self.slowκ].shift(lookback)) &
            (self.data[self.slowκ].shift(1) < self.data[self.slowκ].shift(lookback + 1))
        )
        bearish_divergence = price_higher_high & stoch_lower_high
        
        self.signals['stoch_divergence'] = np.select(
            [bullish_divergence, bearish_divergence],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def stochastic_k_momentum_signals(self,):
        
        """
        Stochastic Momentum Signals
        2 = Stochastic Rising (Bullish Momentum)
        1 = Stochastic Falling (Bearish Momentum)
        0 = Stochastic Stable
        
        """
        
        stoch_rising = self.data[self.slopek] > 0
        stoch_falling = self.data[self.slopek] < 0
        
        self.signals['stoch_k_momentum'] = np.select(
            [stoch_rising, stoch_falling],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def stochastic_d_momentum_signals(self):
        
        """
        Stochastic %D Slope Signals (Signal line momentum)
        2 = %D Rising (Bullish)
        1 = %D Falling (Bearish) 
        0 = %D Stable
        
        """
        
        d_rising = self.data[self.slowd] > 0
        d_falling = self.data[self.sloped] < 0
        
        self.signals['stoch_d_momentum'] = np.select(
            [d_rising, d_falling],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def stochastic_reversal_signals(
        self,
        overbought: int = 80,
        oversold: int = 20
    ):
        
        """
        Stochastic Trend Reversal Signals
        2 = Bullish Reversal (Stochastic exits oversold <20 to >20)
        1 = Bearish Reversal (Stochastic exits overbought >80 to <80)
        0 = No reversal
        
        Parameters:
        overbought (int): Overbought threshold
        oversold (int): Oversold threshold
        
        """
        
        bullish_reversal = (
            (self.data[self.slowκ] > oversold) & 
            (self.data[self.slowκ].shift(1) <= oversold)
        )
        
        bearish_reversal = (
            (self.data[self.slowκ] < overbought) & 
            (self.data[self.slowκ].shift(1) >= overbought)
        )
        
        self.signals['stoch_reversal'] = np.select(
            [bullish_reversal, bearish_reversal],
            [2, 1],
            default=0
        )
        
        return self.signals
    def generate_all_stochastic_signals(self):
        
        """
        Generate all Stochastic signals
        
        """
        
        self.stochastic_overbought_oversold_signals()
        self.stochastic_crossover_signals()
        self.stochastic_divergence_signals()
        self.stochastic_k_momentum_signals()
        self.stochastic_d_momentum_signals()
        self.stochastic_reversal_signals()
        print(self.signals.tail(10), "\n", self.signals.shape)
        return self.signals