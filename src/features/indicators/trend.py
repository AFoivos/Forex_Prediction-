import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')


class ForexTrendIndicators:
    def __init__(self, 
                 data: pd.DataFrame,
                 open_col: str = 'open',
                 high_col: str = 'high', 
                 low_col: str = 'low', 
                 close_col: str = 'close'):
        
        """
        Class for Trend Indicators
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        open_col (str): Column name for open price
        high_col (str): Column name for high price
        low_col (str): Column name for low price
        close_col (str): Column name for close price
        
        """
        
        print("="*50)
        print("TREND INDICATORS")
        print("="*50)
        
        self.data = data.copy()
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.available_get_trend_strength = [False,False,False]
        
        #Validate data_cols
        if not all(col in self.data.columns for col in [self.open_col, self.high_col, self.low_col, self.close_col]):
            raise ValueError ("Invalid column names in DataFrame")
        
    def add_sma(self, 
                periods: List[int] = [10, 20, 50, 100, 200]
                ):
        
        """
        Simple Moving Averages
        
        Parameters:
        periods (List[int]): List of periods for SMA
        
        """ 
        
        print("="*50)
        print("SMA INDICATOR")
        print("="*50)
        
        for period in periods:
            col_name = f'sma_{period}'
            self.data[col_name] = talib.SMA(self.data[self.close_col], timeperiod=period)
            # Slope of SMA
            self.data[f'{col_name}_slope'] = self.data[col_name].diff()
        
        # SMA Signals 
        last_sma = f'sma_{periods[-1]}'
        self.data[f'{last_sma}_signal'] = np.where(
            self.data[self.close_col] > self.data[last_sma], 1, -1
        )
        
        self.available_get_trend_strength[0] = True
        
        print('New columns added: sma_20, sma_50, sma_100, sma_200, sma_20_slope, sma_50_slope, sma_100_slope, sma_200_slope, sma_20_signal, sma_50_signal, sma_100_signal, sma_200_signal')
        print("="*50)
        
        return self.data
        
    def add_ema(self, 
                periods: List[int] = [10, 20, 50, 100, 200]
                ):
        
        """
        Exponential Moving Averages 
        
        Parameters:
        periods (List[int]): List of periods for EMA
        
        """
        
        print("="*50)
        print("EMA INDICATOR")
        print("="*50)
        
        for period in periods:
            col_name = f'ema_{period}'
            self.data[col_name] = talib.EMA(self.data[self.close_col], timeperiod=period)
        
        # EMA signals
        self.data[f'{col_name}_signal'] = np.where(
            self.data[self.close_col] > self.data[col_name], 1, -1
        )
        
        # EMA slope
        self.data[f'{col_name}_slope'] = self.data[col_name].diff()
        
        self.available_get_trend_strength[1] = True
        
        print('New columns added: ema_20, ema_50, ema_100, ema_200, ema_20_slope, ema_50_slope, ema_100_slope, ema_200_slope, ema_20_signal, ema_50_signal, ema_100_signal, ema_200_signal')
        print("="*50)
        
        return self.data
        
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
        print("="*50)
        print("MACD INDICATOR")
        print("="*50)
        
        macd, macd_signal, macd_hist = talib.MACD(
            self.data[self.close_col],
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
        
        self.data['macd_line'] = macd
        self.data['macd_signal'] = macd_signal
        self.data['macd_histogram'] = macd_hist
        
        # MACD signals
        self.data['macd_cross'] = np.where(macd > macd_signal, 1, -1)
        self.data['macd_above_zero'] = (macd > 0).astype(int)
        self.data['macd_signal_above_zero'] = (macd_signal > 0).astype(int)
        
        # MACD histogram changes
        self.data['macd_hist_change'] = macd_hist.diff()
        
        print('New columns added: macd_line, macd_signal, macd_histogram, macd_cross, macd_above_zero, macd_signal_above_zero, macd_hist_change')
        print("="*50)
        
        return self.data
    
    def add_adx(self, period: int = 14):

        """
        Average Directional Movement Index
        
        Parameters:
        period (int): Period for ADX
        
        """
        
        print("="*50)
        print("ADX INDICATOR")
        print("="*50)        

        adx = talib.ADX(
            self.data[self.high_col],
            self.data[self.low_col],
            self.data[self.close_col],
            timeperiod=period
        )
        
        # Positive and Negative Directional Indicators
        plus_di = talib.PLUS_DI(
            self.data[self.high_col],
            self.data[self.low_col],
            self.data[self.close_col],
            timeperiod=period
        )
        
        minus_di = talib.MINUS_DI(
            self.data[self.high_col],
            self.data[self.low_col],
            self.data[self.close_col],
            timeperiod=period
        )
        
        self.data['adx'] = adx
        self.data['plus_di'] = plus_di
        self.data['minus_di'] = minus_di

        # ADX Signals
        self.data['adx_trend_strength'] = pd.cut(
            adx,
            bins=[0, 25, 50, 75, 100],
            labels=['weak', 'moderate', 'strong', 'very_strong']
        )
        
        self.data['adx_strong_trend'] = (adx > 25).astype(int)
        self.data['di_crossover'] = np.where(plus_di > minus_di, 1, -1)
        
        # Trend direction based on DI
        self.data['trend_direction'] = np.where(
            plus_di > minus_di, 
            'uptrend', 
            np.where(plus_di < minus_di, 'downtrend', 'neutral')
        )
        
        self.available_get_trend_strength[2] = True
        
        print('New columns added: adx, plus_di, minus_di, adx_trend_strength, adx_strong_trend, di_crossover, trend_direction')
        print("="*50)
        
        return self.data
        
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
        
        print("="*50)
        print("PARABOLIC SAR INDICATOR")
        print("="*50)
        
        sar = talib.SAR(
            self.data[self.high_col],
            self.data[self.low_col],
            acceleration=acceleration,
            maximum=maximum
        )
        
        self.data['parabolic_sar'] = sar
        
        # SAR signals
        self.data['sar_signal'] = np.where(
            self.data[self.close_col] > sar, 1, -1
        )
        
        self.data['sar_above_price'] = (sar > self.data[self.close_col]).astype(int)
        
        # SAR trend changes
        self.data['sar_trend_change'] = self.data['sar_signal'].diff()
        self.data['sar_trend_change'] = self.data['sar_trend_change'].fillna(0)
        
        print('New columns added: parabolic_sar, sar_signal, sar_above_price, sar_trend_change')
        print("="*50)
        
        return self.data    
    
    def add_trend_confirmation(self):
        
        """
        Adding trend confirmation features
        
        """ 
        
        print("="*50)
        print("TREND CONFIRMATION")
        print("="*50)
        
        # Price above/below key moving averages
        if 'sma_50' in self.data.columns and 'sma_200' in self.data.columns:
            self.data['price_above_sma50'] = (self.data[self.close_col] > self.data['sma_50']).astype(int)
            self.data['price_above_sma200'] = (self.data[self.close_col] > self.data['sma_200']).astype(int)
            self.data['golden_cross'] = ((self.data['sma_50'] > self.data['sma_200']) & (self.data['sma_50'].shift(1) <= self.data['sma_200'].shift(1))).astype(int)
            self.data['death_cross'] = ((self.data['sma_50'] < self.data['sma_200']) & (self.data['sma_50'].shift(1) >= self.data['sma_200'].shift(1))).astype(int)
        
        # Multiple timeframe trend confirmation
        if all(col in self.data.columns for col in ['sma_20', 'sma_50', 'sma_100']):
            self.data['multi_tf_bullish'] = ((self.data[self.close_col] > self.data['sma_20']) & 
                                         (self.data[self.close_col] > self.data['sma_50']) &
                                         (self.data[self.close_col] > self.data['sma_100'])).astype(int)
            
            self.data['multi_tf_bearish'] = ((self.data[self.close_col] < self.data['sma_20']) & 
                                         (self.data[self.close_col] < self.data['sma_50']) &
                                         (self.data[self.close_col] < self.data['sma_100'])).astype(int)
            
        print('New columns added: price_above_sma50, price_above_sma200, golden_cross, death_cross, multi_tf_bullish, multi_tf_bearish')
        print("="*50)
        
        return self.data
    
    def get_trend_strength(self) -> pd.Series:
        
        """
        Returns trend strength score
        
        10: Strong Uptrend
        5: Neutral Trend 
        0: Strong Downtrend
        
        """
        
        print("="*50)
        print("TREND STRENGTH")
        print("="*50)
        
        if self.available_get_trend_strength == [True,True,True]:
            strength_score = pd.Series(5.0, index=self.data.index)  
            
            # ADX contribution
            if 'adx' in self.data.columns:
                # ADX > 25 indicates meaningful trend
                adx_contribution = np.clip((self.data['adx'] - 25) / 25, 0, 3)
                strength_score += adx_contribution
            
            # Moving average slope contribution
            slope_cols = [col for col in self.data.columns if col.endswith('_slope')]
            for col in slope_cols:
                # Normalize slope to reasonable range
                slope_contribution = np.clip(self.data[col] * 100, -2, 2)
                strength_score += slope_contribution
            
            # Ensure final score is between 0-10
            strength_score = np.clip(strength_score, 0, 10)
            strength_score = strength_score.round()  
            current_trend_strength = strength_score.iloc[-1]
            
            print(current_trend_strength)
            
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
        
        return self.data
        

            