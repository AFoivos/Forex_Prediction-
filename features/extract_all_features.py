import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

from .indicators import (
    ForexMomentumIndicators,
    ForexTrendIndicators,
    ForexVolatilityIndicators,
    ForexTSIndicators
)

from .signals import (
    ForexMASignals,
    ForexMACDSignals,
    ForexParabolicSARSignals,
    ForexADXSignals,
    ForexRSISignals,
    ForexStochasticSignals,
    ForexWilliamsRSignals,
    ForexCCISignals,
    ForexMomentumSignals,
    ForexATRSignals,
    ForexBollingerBandsSignals,
    ForexKeltnerSignals,
    ForexSTDSignals,
)

import warnings
warnings.filterwarnings('ignore')

class ForexFeauturesExtractor:
    def __init__(
        self, 
        data: pd.DataFrame,
        momentum_parameters: Dict = {
            'rsi_periods': [14, 21, 28],
            'stoch_fk_sk_sd_periods': [14, 3, 3],
            'williams_periods': [14, 21, 28],
            'cci_periods': [14, 21, 28],
            'momentum_periods': [10, 14, 20]
        },
        trend_parameters: Dict = {
            'sma_periods': [10, 20, 50, 100, 200],
            'ema_periods': [10, 20, 50, 100, 200],
            'macd_fast_slow_signal': [12, 26, 9],
            'adx_periods': [14, 21, 28],
            'sar_acc_max': [0.02, 0.2]   
        },
        volatility_parameters: Dict = {
            'atr_periods': [14, 21, 28],
            'bb_period_nbdevup_nbdevdn': [20, 2.0, 2.0],
            'keltner_ema_atr_multiplier': [20, 10, 2.0],
            'std_periods': [20, 50, 100]
        },        
        open_col: str = 'open',
        high_col: str = 'high', 
        low_col: str = 'low', 
        close_col: str = 'close',
        volume_col: str = 'volume',
    ):
        
        """
        Class for Extracting Forex Indicators & Signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data
        parameters (Dict): Dictionary of parameters for each indicator
        
        """
        
        self.data = data.copy()
        self.indicators_data = pd.DataFrame()
        self.signals_data = pd.DataFrame()
        
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        
        self._validate_columns()
        
        self.momentum_parameters = momentum_parameters
        self.trend_parameters = trend_parameters
        self.volatility_parameters = volatility_parameters
        
        
    def _validate_columns(
        self, 
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
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
       
    def _extract_indicators(
        self
    ):
        
        """
        Extract all indicators
        
        """
        
        momentum_data, self.momentum_parameters = ForexMomentumIndicators(
            data = self.data,
            open_col = self.open_col,
            high_col = self.high_col,
            low_col = self.low_col,
            close_col = self.close_col,
            volume_col = self.volume_col,
            prints = False
            ).generate_all_momentum_indicators(
            **self.momentum_parameters
        )
        print(self.momentum_parameters)
        print(type(momentum_data))
        
        trend_data, self.trend_parameters = ForexTrendIndicators(
            data = self.data,
            open_col = self.open_col,
            high_col = self.high_col,
            low_col = self.low_col,
            close_col = self.close_col,
            volume_col = self.volume_col,
            prints = False
            ).generate_all_trend_indicators(
            **self.trend_parameters
        )
        print(self.trend_parameters)
        print(type(trend_data))
        
        volatility_data, self.volatility_parameters = ForexVolatilityIndicators(
            data = self.data,
            open_col = self.open_col,
            high_col = self.high_col,
            low_col = self.low_col,
            close_col = self.close_col,
            volume_col = self.volume_col,
            prints = False
            ).generate_all_volatility_indicators(
            **self.volatility_parameters
        )
        print(self.volatility_parameters)
        print(type(volatility_data))
        
        self.indicators_data = pd.concat(
            [
                momentum_data,
                trend_data.drop(columns = self.close_col),
                volatility_data.drop(columns = self.close_col)
            ], 
            axis=1,
        )
        print(self.indicators_data)
        print(type(self.indicators_data))
        self.indicators_data.index = self.data.index
        return self.indicators_data

    def _extract_signals(
        self
    ):
        
        """
        Extract all signals
        
        """
        
        print(self.indicators_data)
        rsi_signals = ForexRSISignals(
            data = self.indicators_data,
            close_col = self.close_col,
            parameters = self.momentum_parameters['rsi_params'],
            prints = False
        ).generate_all_rsi_signals()
        
        stochastic_signals = ForexStochasticSignals(
            data = self.indicators_data,
            close_col = self.close_col,
            parameters = self.momentum_parameters['stochastic_params'],
            prints = False
        ).generate_all_stochastic_signals()

        williams_r_signals = ForexWilliamsRSignals(
            data = self.indicators_data,
            close_col = self.close_col,
            parameters = self.momentum_parameters['williams_r_params'],
            prints = False
        ).generate_all_williams_signals()

        cci_signals = ForexCCISignals(
            data = self.indicators_data,
            close_col = self.close_col,
            parameters = self.momentum_parameters['cci_params'],
            prints = False
        ).generate_all_cci_signals()

        momentum_signals = ForexMomentumSignals(
            data = self.indicators_data,
            close_col = self.close_col,
            parameters = self.momentum_parameters['momentum_params'],
            prints = False
        ).generate_all_momentum_signals()
        
        mas_signals = ForexMASignals(
            data = self.indicators_data,
            close_col = self.close_col,
            ema_parameters=self.trend_parameters['ema_params'],
            sma_parameters=self.trend_parameters['sma_params'],
            prints = False
        ).generate_all_signals()
        
        macd_signals = ForexMACDSignals(
            data = self.indicators_data,
            close_col = self.close_col,
            parameters = self.trend_parameters['macd_params'],
            prints = False
        ).generate_all_macd_signals()   
        
        adx_signals = ForexADXSignals(
            data = self.indicators_data,
            close_col = self.close_col,
            parameters = self.trend_parameters['adx_params'],
            prints = False
        ).generate_all_adx_signals()
        
        sar_signals = ForexParabolicSARSignals(
            data = self.indicators_data,
            close_col = self.close_col,
            parameters = self.trend_parameters['sar_params'],
            prints = False
        ).generate_all_sar_signals()
        
        atr_signals = ForexATRSignals(
            data = self.indicators_data,
            close_col = self.close_col,
            parameters = self.volatility_parameters['atr_params'],
            prints = False
        ).generate_all_atr_signals()
        
        bb_signals = ForexBollingerBandsSignals(
            data = self.indicators_data,
            close_col = self.close_col,
            parameters = self.volatility_parameters['bb_params'],
            prints = False
        ).generate_all_bb_signals()
        
        keltner_signals = ForexKeltnerSignals(
            data = self.indicators_data,
            close_col = self.close_col,
            parameters = self.volatility_parameters['keltner_params'],
            prints = False
        ).generate_all_keltner_signals()
        
        std_signals = ForexSTDSignals(
            data = self.indicators_data,
            close_col = self.close_col,
            parameters = self.volatility_parameters['std_params'],
            prints = False
        ).generate_all_std_signals()
        
        self.signals_data = pd.concat(
            [
                rsi_signals,
                stochastic_signals.drop(columns = self.close_col),
                williams_r_signals.drop(columns = self.close_col),
                cci_signals.drop(columns = self.close_col),
                momentum_signals.drop(columns = self.close_col),
                mas_signals.drop(columns = self.close_col),
                macd_signals.drop(columns = self.close_col),
                adx_signals.drop(columns = self.close_col),
                sar_signals.drop(columns = self.close_col),
                atr_signals.drop(columns = self.close_col),
                bb_signals.drop(columns = self.close_col),
                keltner_signals.drop(columns = self.close_col),
                std_signals.drop(columns = self.close_col)
            ],
            axis=1
        )
        
        return self.signals_data 
    
    def extract_all_features(
        self
    ):
        
        """
        Extract all features
        
        """
        
        self._extract_indicators()
        self._extract_signals()
        
        data_with_signals_and_indicators = pd.concat(
            [
                self.indicators_data,
                self.signals_data.drop(columns = self.close_col)
            ],
            axis=1
        )
        
        return self.indicators_data, self.signals_data, data_with_signals_and_indicators
                    
                        
        
        
        
        