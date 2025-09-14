import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')


class TrendIndicators:
    def __init__(self, 
                 df: pd.DataFrame,
                 open_col: str = 'open',
                 high_col: str = 'high', 
                 low_col: str = 'low', 
                 close_col: str = 'close'):
        
        """
        Class for Trend Indicators
        
        Parameters:
        df (pd.DataFrame): DataFrame containing the data    
        open_col (str): Column name for open price
        high_col (str): Column name for high price
        low_col (str): Column name for low price
        close_col (str): Column name for close price
        
        """
        
        self.df = df.copy()
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.available_get_trend_strength = [False,False,False]
        
        #Validate df_cols
        if not all(col in self.df.columns for col in [self.open_col, self.high_col, self.low_col, self.close_col]):
            raise ValueError ("Invalid column names in DataFrame")
        
    def add_sma(self, 
                periods: List[int] = [20, 50, 100]
                ):
        
        """
        Simple Moving Averages
        
        Parameters:
        periods (List[int]): List of periods for SMA
        
        """ 
        
        for period in periods:
            col_name = f'sma_{period}'
            self.df[col_name] = talib.SMA(self.df[self.close_col], timeperiod=period)
            # Slope of SMA
            self.df[f'{col_name}_slope'] = self.df[col_name].diff()
        
        # SMA Signals 
        last_sma = f'sma_{periods[-1]}'
        self.df[f'{last_sma}_signal'] = np.where(
            self.df[self.close_col] > self.df[last_sma], 1, -1
        )
        
        self.available_get_trend_strength[0] = True
        
    def add_ema(self, 
                periods: List[int] = [12, 26, 50]
                ):
        
        """
        Exponential Moving Averages 
        
        Parameters:
        periods (List[int]): List of periods for EMA
        
        """
        
        for period in periods:
            col_name = f'ema_{period}'
            self.df[col_name] = talib.EMA(self.df[self.close_col], timeperiod=period)
        
        # EMA signals
        self.df[f'{col_name}_signal'] = np.where(
            self.df[self.close_col] > self.df[col_name], 1, -1
        )
        
        # EMA slope
        self.df[f'{col_name}_slope'] = self.df[col_name].diff()
        
        self.available_get_trend_strength[1] = True
        
    def add_macd(self, 
                    fastperiod: int = 12,
                    slowperiod: int = 26,
                    signalperiod: int = 9
                    ):
        
        """
        MACD Indicator 
        
        Parameters:
        fastperiod (int): Fast period for MACD
        slowperiod (int): Slow period for MACD
        signalperiod (int): Signal period for MACD
        
        """
        
        macd, macd_signal, macd_hist = talib.MACD(
            self.df[self.close_col],
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
        
        self.df['macd_line'] = macd
        self.df['macd_signal'] = macd_signal
        self.df['macd_histogram'] = macd_hist
        
        # MACD signals
        self.df['macd_cross'] = np.where(macd > macd_signal, 1, -1)
        self.df['macd_above_zero'] = (macd > 0).astype(int)
        self.df['macd_signal_above_zero'] = (macd_signal > 0).astype(int)
        
        # MACD histogram changes
        self.df['macd_hist_change'] = macd_hist.diff()

        
    def add_adx(self, 
                period: int = 14
                ):

        """
        Average Directional Movement Index
        
        Parameters:
        period (int): Period for ADX
        
        """
        adx = talib.ADX(
            self.df[self.high_col],
            self.df[self.low_col],
            self.df[self.close_col],
            timeperiod=period
        )
        
        # Positive and Negative Directional Indicators
        plus_di = talib.PLUS_DI(
            self.df[self.high_col],
            self.df[self.low_col],
            self.df[self.close_col],
            timeperiod=period
        )
        
        minus_di = talib.MINUS_DI(
            self.df[self.high_col],
            self.df[self.low_col],
            self.df[self.close_col],
            timeperiod=period
        )
        
        self.df['adx'] = adx
        self.df['plus_di'] = plus_di
        self.df['minus_di'] = minus_di

        # ADX Signals
        self.df['adx_trend_strength'] = pd.cut(
            adx,
            bins=[0, 25, 50, 75, 100],
            labels=['weak', 'moderate', 'strong', 'very_strong']
        )
        
        self.df['adx_strong_trend'] = (adx > 25).astype(int)
        self.df['di_crossover'] = np.where(plus_di > minus_di, 1, -1)
        
        # Trend direction based on DI
        self.df['trend_direction'] = np.where(
            plus_di > minus_di, 
            'uptrend', 
            np.where(plus_di < minus_di, 'downtrend', 'neutral')
        )
        
        self.available_get_trend_strength[2] = True
            
    def add_parabolic_sar(self, 
                          acceleration: float = 0.02,
                          maximum: float = 0.2
                          ):
        
        """
        Parabolic Stop and Reverse
        
        Parameters:
        acceleration (float): Acceleration parameter for SAR
        maximum (float): Maximum parameter for SAR
        
        """
        sar = talib.SAR(
            self.df[self.high_col],
            self.df[self.low_col],
            acceleration=acceleration,
            maximum=maximum
        )
        
        self.df['parabolic_sar'] = sar
        
        # SAR signals
        self.df['sar_signal'] = np.where(
            self.df[self.close_col] > sar, 1, -1
        )
        
        self.df['sar_above_price'] = (sar > self.df[self.close_col]).astype(int)
        
        # SAR trend changes
        self.df['sar_trend_change'] = self.df['sar_signal'].diff()
        self.df['sar_trend_change'] = self.df['sar_trend_change'].fillna(0)
    
    def add_trend_confirmation(self):
        
        """
        Adding trend confirmation features
        
        """ 
        
        # Price above/below key moving averages
        if 'sma_50' in self.df.columns and 'sma_200' in self.df.columns:
            
            self.df['price_above_sma50'] = (self.df[self.close_col] > self.df['sma_50']).astype(int)
            
            self.df['price_above_sma200'] = (self.df[self.close_col] > self.df['sma_200']).astype(int)
            
            self.df['golden_cross'] = ((self.df['sma_50'] > self.df['sma_200']) & (self.df['sma_50'].shift(1) <= self.df['sma_200'].shift(1))).astype(int)
            
            self.df['death_cross'] = ((self.df['sma_50'] < self.df['sma_200']) & (self.df['sma_50'].shift(1) >= self.df['sma_200'].shift(1))).astype(int)
        
        # Multiple timeframe trend confirmation
        if all(col in self.df.columns for col in ['sma_20', 'sma_50', 'sma_100']):
            self.df['multi_tf_bullish'] = ((self.df[self.close_col] > self.df['sma_20']) & 
                                         (self.df[self.close_col] > self.df['sma_50']) &
                                         (self.df[self.close_col] > self.df['sma_100'])).astype(int)
            
            self.df['multi_tf_bearish'] = ((self.df[self.close_col] < self.df['sma_20']) & 
                                         (self.df[self.close_col] < self.df['sma_50']) &
                                         (self.df[self.close_col] < self.df['sma_100'])).astype(int)
        
        return self.df
    
    
    def get_trend_strength(self) -> pd.Series:
        
        """
        Returns trend strength score
        
        10: Strong Uptrend
        5: Neutral Trend 
        0: Strong Downtrend
        
        """
        if self.available_get_trend_strength == [True,True,True]:
            strength_score = pd.Series(5.0, index=self.df.index)  
            
            # ADX contribution
            if 'adx' in self.df.columns:
                # ADX > 25 indicates meaningful trend
                adx_contribution = np.clip((self.df['adx'] - 25) / 25, 0, 3)
                strength_score += adx_contribution
            
            # Moving average slope contribution
            slope_cols = [col for col in self.df.columns if col.endswith('_slope')]
            for col in slope_cols:
                # Normalize slope to reasonable range
                slope_contribution = np.clip(self.df[col] * 100, -2, 2)
                strength_score += slope_contribution
            
            # Ensure final score is between 0-10
            strength_score = np.clip(strength_score, 0, 10)
            strength_score = strength_score.round()  
            current_trend_strength = strength_score.iloc[-1]
            
            return current_trend_strength
        else:
            print('You need to calculate EMA,SMA and ADX before you can get trend strength')
         
    def get_all_trend_indicators(self,
                                    sma_periods: List[int] = [20, 50, 100],
                                    ema_periods: List[int] = [12, 26, 50],
                                    macd_fastperiod: int = 12,
                                    macd_slowperiod: int = 26,
                                    macd_signalperiod: int = 9,
                                    adx_period: int = 14,
                                    parabolic_sar_acceleration: float = 0.02,
                                    parabolic_sar_maximum: float = 0.2
                                    ):
        
        """
        Adds all trend indicators
        
        Parameters:
        sma_periods (List[int]): List of periods for SMA
        ema_periods (List[int]): List of periods for EMA
        macd_fastperiod (int): Fast period for MACD
        macd_slowperiod (int): Slow period for MACD
        macd_signalperiod (int): Signal period for MAC  
        adx_period (int): Period for ADX
        parabolic_sar_acceleration (float): Acceleration parameter for SAR
        parabolic_sar_maximum (float): Maximum parameter for SAR        
        
        """
        
        self.add_sma(sma_periods)
        self.add_ema(ema_periods)
        self.add_macd(macd_fastperiod, macd_slowperiod, macd_signalperiod)
        self.add_adx(adx_period)
        self.add_parabolic_sar(parabolic_sar_acceleration, parabolic_sar_maximum)
        self.add_trend_confirmation()
        self.get_trend_strength()
        
        return self.df
        

            