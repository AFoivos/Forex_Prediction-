import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexPriceBasedLabelGenerator:
    def __init__(
        self,
        data: pd.DataFrame, 
        price_col: str ='close',
        open_col: str = 'open',
        high_col: str = 'high', 
        low_col: str = 'low', 
        close_col: str = 'close',
        volume_col: str = 'volume',
        volatility_window: int = 20
    ):
    
        """
        Class for Label Generation

        Parameters:             
        data (pd.DataFrame): DataFrame containing the data  
        price_col (str): Column name for close price
        volatility_window (int): Rolling window for volatility calculation
        
        Optional: (if you dont have them we will calculate them for you)
        Average True Range (atr)
        volatility
        returns 
        
        """
        
        print("="*50)
        print("LABEL GENERATION")
        print("="*50)
        print(" Available Functions: \n1 generate_directional_labels \n2 generate_return_threshold_labels \n3 generate_time_horizon_labels \n4 generate_all_labels \n5 get_label_columns")
        print("="*50)
    
        self.data = data.copy() 
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col  
        self.price_col = price_col
        self.volatility_window = volatility_window
        
        # Validate needed columns exist
        self._validate_data_cols()
    
    def _validate_data_cols(self):
        
        """
        Pre-calculate only MISSING metrics
        
        """
        
        required_metrics = {
            'returns': ['ret_1h', 'ret_4h', 'ret_24h', 'ret_120h'],
            'volatility': ['vol_1h', 'vol_4h'],
            'atr': ['atr'] 
        }
        
        missing_metrics = []
        
        # Check returns
        for ret_col in required_metrics['returns']:
            if ret_col not in self.data.columns:
                missing_metrics.append(ret_col)
        
        # Check volatility
        for vol_col in required_metrics['volatility']:
            if vol_col not in self.data.columns:
                missing_metrics.append(vol_col)
        
        # Check ATR - look for any column starting with 'atr'
        atr_columns = [col for col in self.data.columns if col.startswith('atr')]
        
        if not atr_columns:
            missing_metrics.append('atr (or any column starting with "atr")')
        else:
            if 'atr_14' in atr_columns:
                self.data['atr'] = self.data['atr_14']
            else:
                self.data['atr'] = self.data[atr_columns[0]]
                
        if missing_metrics:
            raise ValueError(f"Missing metrics: {missing_metrics}")
        

    def generate_directional_labels(self):
        
        """
        Directional labels for different timeframes
        
        """
        
        # Check if required returns exist, if not calculate them
        required_returns = ['ret_1h', 'ret_4h', 'ret_daily', 'ret_weekly']
        for ret_col in required_returns:
            if ret_col not in self.data.columns:
                # Calculate missing return
                periods = 1 if ret_col == 'ret_1h' else 4 if ret_col == 'ret_4h' else 24 if ret_col == 'ret_daily' else 120
                self.data[ret_col] = self.data[self.price_col].pct_change(periods)
        
        # 1-hour direction
        self.data['label_dir_1h'] = np.where(self.data['ret_1h'].shift(-1) > 0, 1, -1)
        
        # 4-hour direction
        self.data['label_dir_4h'] = np.where(self.data['ret_4h'].shift(-4) > 0, 1, -1)
        
        # Daily direction  
        self.data['label_dir_24h'] = np.where(self.data['ret_24h'].shift(-24) > 0, 1, -1)
        
        # Weekly direction
        self.data['label_dir_120h'] = np.where(self.data['ret_120h'].shift(-120) > 0, 1, -1)
        
        return self
    
    def generate_return_threshold_labels(self):
        
        """
        Labels based on return thresholds
        
        """
        
        # Ensure ret_1h exists
        fut_ret_1h = self.data['ret_1h'].shift(-1)
        
        # Fixed thresholds
        self.data['label_ret_0.5pct_1h'] = np.where(abs(fut_ret_1h) > 0.005, np.sign(fut_ret_1h), 0)
        self.data['label_ret_1pct_1h'] = np.where(abs(fut_ret_1h) > 0.01, np.sign(fut_ret_1h), 0)
        self.data['label_ret_2pct_1h'] = np.where(abs(fut_ret_1h) > 0.02, np.sign(fut_ret_1h), 0)
        
        # Volatility adjusted - ensure vol_1h exists
        vol_adj_1h = 0.02 * (self.data['vol_1h'] / self.data['vol_1h'].median())
        self.data['label_ret_2pct_vol_adj_1h'] = np.where(abs(fut_ret_1h) > vol_adj_1h, np.sign(fut_ret_1h), 0)
        
        return self
    
    def generate_time_horizon_labels(self):
        
        """
        Labels for different trading styles
        
        """
        
        # Scalping (5-15 min)
        scalping_ret = self.data[self.price_col].pct_change(1)
        self.data['label_scalping'] = np.where(
            abs(scalping_ret.shift(-1)) > 0.001, 
            np.sign(scalping_ret.shift(-1)),
            0
        )
        
        # Intraday (1-4 hours) - ensure ret_4h exists
        if 'ret_4h' not in self.data.columns:
            self.data['ret_4h'] = self.data[self.price_col].pct_change(4)
        
        self.data['label_intraday'] = np.where(
            abs(self.data['ret_4h'].shift(-4)) > 0.003,
            np.sign(self.data['ret_4h'].shift(-4)),
            0
        )
        
        # Swing (1-5 days) - ensure ret_daily exists
        if 'ret_daily' not in self.data.columns:
            self.data['ret_daily'] = self.data[self.price_col].pct_change(24)
        
        self.data['label_swing'] = np.where(
            abs(self.data['ret_daily'].shift(-24)) > 0.008,
            np.sign(self.data['ret_daily'].shift(-24)), 
            0
        )
        
        # Use rolling quantiles instead of global quantiles
        self.data['label_scalping_conf'] = self._calculate_rolling_confidence('vol_1h', window=100)
        
        return self
    
    def _calculate_rolling_confidence(self, vol_col, window=100):
        
        """
        Calculate confidence using rolling quantiles (no look-ahead bias)
        
        """
        
        # Calculate rolling quantiles
        rolling_q30 = self.data[vol_col].rolling(window=window, min_periods=window).quantile(0.3)
        rolling_q70 = self.data[vol_col].rolling(window=window, min_periods=window).quantile(0.7)
        
        # Assign confidence levels
        confidence = np.zeros(len(self.data))
        confidence = np.where(self.data[vol_col] < rolling_q30, 2, confidence)  # High
        confidence = np.where((self.data[vol_col] >= rolling_q30) & 
                             (self.data[vol_col] < rolling_q70), 1, confidence)  # Medium
        
        # Set to 0 for first 'window' periods where we don't have enough data
        confidence[:window] = 0
        
        return confidence

    def generate_all_labels(self):
        
        """
        Generate all price-based labels directly into the dataframe
        
        """
        
        print("Generating directional labels...")
        self.generate_directional_labels()
        
        print("Generating return threshold labels...")  
        self.generate_return_threshold_labels()
        
        print("Generating time horizon labels...")
        self.generate_time_horizon_labels()
        
        # Get all label columns created
        label_columns = [col for col in self.data.columns if col.startswith('label_')]
        print(f"Generated {len(label_columns)} labels directly in dataframe")
        
        return self.data
    
   