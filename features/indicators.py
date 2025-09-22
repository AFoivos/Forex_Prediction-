import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexMomentumIndicators:
    def __init__(self, 
                 data: pd.DataFrame,
                 open_col: str = 'open',
                 high_col: str = 'high', 
                 low_col: str = 'low', 
                 close_col: str = 'close',
                 volume_col: str = 'volume'):
        
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
        print(" Available Fuctions \n1 add_rsi \n2 add_stochastic \n3 add_williams_r \n4 add_cci \n5 add_momentum \n6 add_momentum_confirmation")
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
    
    def add_rsi(self, periods: List[int] = [14, 21, 28]):
        
        """
        Relative Strength Index
        
        Parameters:
        periods (List[int]): List of periods for RSI
        
        """
                
        for period in periods:
            col_name = f'rsi_{period}'
            self.data[col_name] = talib.RSI(self.data[self.close_col], timeperiod=period)
            
            # RSI signals
            self.data[f'{col_name}_overbought'] = (self.data[col_name] > overbought).astype(int)
            self.data[f'{col_name}_oversold'] = (self.data[col_name] < oversold).astype(int)
            self.data[f'{col_name}_signal'] = np.where(
                self.data[col_name] > 50, 1, -1
            )
            
            # RSI divergence
            self.data[f'{col_name}_trend'] = self.data[col_name].diff()
        
        return self.data
    
    def add_stochastic(self,
                      fastk_period: int = 14,
                      slowk_period: int = 3,
                      slowd_period: int = 3,
                      overbought: int = 80,
                      oversold: int = 20):
        
        """
        Stochastic Oscillator
        
        Parameters:
        fastk_period (int): Fast %K period
        slowk_period (int): Slow %K period
        slowd_period (int): Slow %D period
        overbought (int): Overbought threshold
        oversold (int): Oversold threshold
        
        """
        
        slowk, slowd = talib.STOCH(
            self.data[self.high_col],
            self.data[self.low_col],
            self.data[self.close_col],
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=0,
            slowd_period=slowd_period,
            slowd_matype=0
        )
        
        stochastic_columns = [
            'stoch_slowk', 'stoch_slowd', 'stoch_crossover',
            'stoch_overbought', 'stoch_oversold', 'stoch_signal'
        ]
        
        self.data['stoch_slowk'] = slowk
        self.data['stoch_slowd'] = slowd
        
        # Stochastic signals
        self.data['stoch_crossover'] = np.where(slowk > slowd, 1, -1)
        self.data['stoch_overbought'] = ((slowk > overbought) & (slowd > overbought)).astype(int)
        self.data['stoch_oversold'] = ((slowk < oversold) & (slowd < oversold)).astype(int)
        self.data['stoch_signal'] = np.where(
            (slowk < oversold) & (slowd < oversold), 1,
            np.where((slowk > overbought) & (slowd > overbought), -1, 0)
        )
                
        return self.data
    
    def add_williams_r(self,
                      period: int = 14,
                      overbought: int = -20,
                      oversold: int = -80):
        
        """
        Williams %R
        
        Parameters:
        period (int): Period for Williams %R
        overbought (int): Overbought threshold
        oversold (int): Oversold threshold
        
        """
        
        willr = talib.WILLR(
            self.data[self.high_col],
            self.data[self.low_col],
            self.data[self.close_col],
            timeperiod=period
        )
        
        willr_columns = [
            'williams_r', 'willr_overbought', 'willr_oversold',
            'willr_signal', 'willr_trend'
        ]
        
        self.data['williams_r'] = willr
        
        # Williams %R signals
        self.data['willr_overbought'] = (willr > overbought).astype(int)
        self.data['willr_oversold'] = (willr < oversold).astype(int)
        self.data['willr_signal'] = np.where(
            willr < oversold, 1,
            np.where(willr > overbought, -1, 0)
        )
        self.data['willr_trend'] = willr.diff()
                
        return self.data
    
    def add_cci(self,
               period: int = 20,
               overbought: int = 100,
               oversold: int = -100):
        
        """
        Commodity Channel Index
        
        Parameters:
        period (int): Period for CCI
        overbought (int): Overbought threshold
        oversold (int): Oversold threshold
        
        """
        
        cci = talib.CCI(
            self.data[self.high_col],
            self.data[self.low_col],
            self.data[self.close_col],
            timeperiod=period
        )
        
        cci_columns = [
            'cci', 'cci_overbought', 'cci_oversold',
            'cci_signal', 'cci_trend'
        ]
        
        self.data['cci'] = cci
        
        # CCI signals
        self.data['cci_overbought'] = (cci > overbought).astype(int)
        self.data['cci_oversold'] = (cci < oversold).astype(int)
        self.data['cci_signal'] = np.where(
            cci > 0, 1, -1
        )
        self.data['cci_trend'] = cci.diff()
                
        return self.data
    
    def add_momentum(self, periods: List[int] = [10, 14, 20]):
        
        """
        Momentum Indicator
        
        Parameters:
        periods (List[int]): List of periods for Momentum
        
        """
        
        for period in periods:
            col_name = f'momentum_{period}'
            self.data[col_name] = talib.MOM(self.data[self.close_col], timeperiod=period)
            
            # Momentum signals
            self.data[f'{col_name}_signal'] = np.where(
                self.data[col_name] > 0, 1, -1
            )
            self.data[f'{col_name}_strength'] = self.data[col_name].abs()
            self.data[f'{col_name}_trend'] = self.data[col_name].diff()
        
        return self.data
    
    def get_all_momentum_indicators(self,
                                   rsi_periods: List[int] = [14, 21, 28],
                                   stochastic_fastk: int = 14,
                                   stochastic_slowk: int = 3,
                                   stochastic_slowd: int = 3,
                                   williams_period: int = 14,
                                   cci_period: int = 20,
                                   momentum_periods: List[int] = [10, 14, 20]):
        
        """
        Adds all momentum indicators
        
        Parameters:
        rsi_periods (List[int]): List of periods for RSI
        stochastic_fastk (int): Fast %K period for Stochastic
        stochastic_slowk (int): Slow %K period for Stochastic
        stochastic_slowd (int): Slow %D period for Stochastic
        williams_period (int): Period for Williams %R
        cci_period (int): Period for CCI
        momentum_periods (List[int]): List of periods for Momentum
        
        """
        
        self.add_rsi(rsi_periods)
        self.add_stochastic(stochastic_fastk, stochastic_slowk, stochastic_slowd)
        self.add_williams_r(williams_period)
        self.add_cci(cci_period)
        self.add_momentum(momentum_periods)
        
        return self.data
    
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
        
        return self.data
    
    def add_volume_sma(self, periods: List[int] = [5, 10, 20, 50]):
        
        """
        Volume Simple Moving Averages
        
        Parameters:
        periods (List[int]): List of periods for Volume SMA
        
        """
        
        for period in periods:
            col_name = f'volume_sma_{period}'
            self.data[col_name] = self.data[self.volume_col].rolling(window=period).mean()
            
            # Volume SMA ratios and signals
            self.data[f'{col_name}_ratio'] = self.data[self.volume_col] / self.data[col_name]
            self.data[f'{col_name}_signal'] = np.where(
                self.data[self.volume_col] > self.data[col_name], 1, -1
            )
            self.data[f'{col_name}_trend'] = self.data[col_name].diff()
            
        return self.data
    
    def add_volume_roc(self, periods: List[int] = [5, 10, 14, 21]):
        
        """
        Volume Rate of Change
        
        Parameters:
        periods (List[int]): List of periods for Volume ROC
        
        """
            
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

        return self.data
    
class ForexVolatilityIndicators:
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
                
        return self.data
    
    def add_standard_deviation(self, periods: List[int] = [20, 50, 100]):
        
        """
        Standard Deviation
        
        Parameters:
        periods (List[int]): List of periods for Standard Deviation
        
        """
        
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
        
        return self.data
    
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
        print(" Available Fuctions: \n1 add_sma \n2 add_ema \n3 add_macd \n4 add_adx \n5 add_parabolic_sar \n6 add_trend_confirmation")
        print("="*50)
        
        self.data = data.copy()
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.available_get_trend_strength = [False,False,False]
        
        #Validate data_cols
        required_cols = [self.open_col, self.high_col, self.low_col, self.close_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        
    def add_sma(self, 
                periods: List[int] = [10, 20, 50, 100, 200]
                ):
        
        """
        Simple Moving Averages
        
        Parameters:
        periods (List[int]): List of periods for SMA
        
        """ 
        
        for period in periods:
            col_name = f'sma_{period}'
            self.data[col_name] = talib.SMA(self.data[self.close_col], timeperiod=period)
            # Slope of SMA
            self.data[f'{col_name}_slope'] = self.data[col_name].diff()
        
        return self.data
        
    def add_ema(self, 
                periods: List[int] = [10, 20, 50, 100, 200]
                ):
        
        """
        Exponential Moving Averages 
        
        Parameters:
        periods (List[int]): List of periods for EMA
        
        """
        
        for period in periods:
            col_name = f'ema_{period}'
            self.data[col_name] = talib.EMA(self.data[self.close_col], timeperiod=period)
        
        # EMA slope
        self.data[f'{col_name}_slope'] = self.data[col_name].diff()
        
        self.available_get_trend_strength[1] = True

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
        
        macd, macd_signal, macd_hist = talib.MACD(
            self.data[self.close_col],
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
        
        self.data['macd_line'] = macd
        self.data['macd_signal'] = macd_signal
        self.data['macd_histogram'] = macd_hist
        
        return self.data
    
    def add_adx(self, period: int = 14):

        """
        Average Directional Movement Index
        
        Parameters:
        period (int): Period for ADX
        
        """  

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
        
        sar = talib.SAR(
            self.data[self.high_col],
            self.data[self.low_col],
            acceleration=acceleration,
            maximum=maximum
        )
        
        self.data['parabolic_sar'] = sar
                   
        return self.data     
         
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
        
        return self.data
    
