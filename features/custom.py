import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexCustomFeatures:
    def __init__(self, 
                 data: pd.DataFrame,
                 open_col: str = 'open',
                 high_col: str = 'high', 
                 low_col: str = 'low', 
                 close_col: str = 'close',
                 volume_col: str = 'volume'):
        
        """
        Class for Custom Features
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        open_col (str): Column name for open price
        high_col (str): Column name for high price
        low_col (str): Column name for low price
        close_col (str): Column name for close price
        volume_col (str): Column name for volume
        
        """
        
        print("="*50)
        print("CUSTOM FEATURES")
        print("="*50)
        print(" Available Fuctions \n1 add_returns_features \n2 add_volatility_measures \n3 add_price_position_features \n4 add_seasonality_features \n5 add_time_based_features \n6 add_custom_derived_features")
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
    
    def add_returns_features(self, periods: List[int] = [1, 5, 10, 20]):
        
        """
        Returns and Logarithmic Returns
        
        Parameters:
        periods (List[int]): List of periods for returns calculation
        
        """
                
        for period in periods:
            # Simple returns
            ret_col = f'return_{period}'
            self.data[ret_col] = self.data[self.close_col].pct_change(period)
            
            # Logarithmic returns
            log_ret_col = f'log_return_{period}'
            self.data[log_ret_col] = np.log(self.data[self.close_col] / self.data[self.close_col].shift(period))
            
            # Returns volatility (rolling std)
            vol_col = f'return_vol_{period}'
            self.data[vol_col] = self.data[ret_col].rolling(window=period).std()
            
            # Returns direction
            dir_col = f'return_direction_{period}'
            self.data[dir_col] = np.where(self.data[ret_col] > 0, 1, -1)
                    
        # Cumulative returns
        self.data['cumulative_return'] = (1 + self.data['return_1']).cumprod() - 1
        self.data['cumulative_log_return'] = self.data['log_return_1'].cumsum()
        
        return self.data
    
    def add_volatility_measures(self, periods: List[int] = [5, 10, 20, 50]):
        
        """
        Advanced Volatility Measures
        
        Parameters:
        periods (List[int]): List of periods for volatility calculation
        
        """
        
        
        for period in periods:
            # Historical volatility (annualized)
            hv_col = f'historical_vol_{period}'
            daily_vol = self.data['log_return_1'].rolling(window=period).std()
            self.data[hv_col] = daily_vol * np.sqrt(252)  # Annualized
            
            # Parkinson volatility (high-low based)
            parkinson_col = f'parkinson_vol_{period}'
            log_hl = np.log(self.data[self.high_col] / self.data[self.low_col])
            self.data[parkinson_col] = np.sqrt((1/(4 * period * np.log(2))) * (log_hl**2).rolling(window=period).sum())
            
            # Garman-Klass volatility
            gk_col = f'garman_klass_vol_{period}'
            log_hl_sq = (np.log(self.data[self.high_col] / self.data[self.low_col]))**2
            log_co_sq = (2 * np.log(2) - 1) * (np.log(self.data[self.close_col] / self.data[self.open_col]))**2
            self.data[gk_col] = np.sqrt((1/period) * (log_hl_sq - log_co_sq).rolling(window=period).sum())
            
            # Volatility ratio (close-to-close vs high-low)
            vol_ratio_col = f'vol_ratio_{period}'
            self.data[vol_ratio_col] = self.data[parkinson_col] / (self.data[hv_col] + 1e-10)
                    
        # Volatility regimes
        self.data['high_volatility_regime'] = (self.data['historical_vol_20'] > self.data['historical_vol_20'].quantile(0.75)).astype(int)
        self.data['low_volatility_regime'] = (self.data['historical_vol_20'] < self.data['historical_vol_20'].quantile(0.25)).astype(int)
        
        return self.data
    
    def add_price_position_features(self):
        
        """
        Price Position in Range Features
        
        """
                
        # Daily price range position
        self.data['daily_range_pct'] = (
            (self.data[self.close_col] - self.data[self.low_col]) / 
            (self.data[self.high_col] - self.data[self.low_col] + 1e-10)
        ) * 100
        
        # Weekly range position (5-day rolling)
        self.data['weekly_high'] = self.data[self.high_col].rolling(window=5).max()
        self.data['weekly_low'] = self.data[self.low_col].rolling(window=5).min()
        self.data['weekly_range_pct'] = (
            (self.data[self.close_col] - self.data['weekly_low']) / 
            (self.data['weekly_high'] - self.data['weekly_low'] + 1e-10)
        ) * 100
        
        # Monthly range position (20-day rolling)
        self.data['monthly_high'] = self.data[self.high_col].rolling(window=20).max()
        self.data['monthly_low'] = self.data[self.low_col].rolling(window=20).min()
        self.data['monthly_range_pct'] = (
            (self.data[self.close_col] - self.data['monthly_low']) / 
            (self.data['monthly_high'] - self.data['monthly_low'] + 1e-10)
        ) * 100
        
        # Range position signals
        self.data['at_daily_high'] = (self.data[self.close_col] == self.data[self.high_col]).astype(int)
        self.data['at_daily_low'] = (self.data[self.close_col] == self.data[self.low_col]).astype(int)
        
        self.data['near_weekly_high'] = (self.data['weekly_range_pct'] > 80).astype(int)
        self.data['near_weekly_low'] = (self.data['weekly_range_pct'] < 20).astype(int)
        
        return self.data
    
    def add_seasonality_features(self):
        
        """
        Seasonality and Calendar Features
        
        """
        
        if not self.has_datetime_index:
            print("⏭️  Skipping Seasonality Features - Index is not datetime")
            return self.data
                
        # Time-based features
        self.data['hour'] = self.data.index.hour
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['day_of_month'] = self.data.index.day
        self.data['week_of_year'] = self.data.index.isocalendar().week
        self.data['month'] = self.data.index.month
        self.data['quarter'] = self.data.index.quarter
        self.data['year'] = self.data.index.year
        
        # Time of day segments
        self.data['time_of_day'] = pd.cut(
            self.data['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        # Day type
        self.data['is_weekend'] = (self.data['day_of_week'] >= 5).astype(int)
        self.data['is_month_end'] = (self.data.index.is_month_end).astype(int)
        self.data['is_quarter_end'] = (self.data.index.is_quarter_end).astype(int)
        self.data['is_year_end'] = (self.data.index.is_year_end).astype(int)
        
        # Seasonal indicators
        self.data['season'] = pd.cut(
            self.data['month'],
            bins=[0, 3, 6, 9, 12],
            labels=['winter', 'spring', 'summer', 'fall'],
            include_lowest=True
        )
        
        return self.data
    
    def add_time_based_features(self):
        
        """
        Advanced Time-based Features
        
        """
        
        # Time since market open (assuming 24/5 market)
        self.data['minutes_since_monday_open'] = (
            (self.data.index.dayofweek * 24 * 60) + 
            (self.data.index.hour * 60) + 
            self.data.index.minute
        )
        
        # Period of day (for intraday patterns)
        self.data['period_of_day'] = pd.cut(
            self.data.index.hour,
            bins=[0, 4, 8, 12, 16, 20, 24],
            labels=['late_night', 'early_morning', 'morning', 'afternoon', 'evening', 'night'],
            include_lowest=True
        )
        
        # Holiday proximity (simplified)
        self.data['days_to_month_end'] = (self.data.index + pd.offsets.MonthEnd(0)).day - self.data.index.day
        self.data['days_to_quarter_end'] = (self.data.index + pd.offsets.QuarterEnd(0)).day - self.data.index.day
        
        return self.data
    
    def add_custom_derived_features(self):
        
        """
        Custom Derived Features
        
        """
                
        # Price momentum features
        if all(col in self.data.columns for col in [self.close_col, 'return_5', 'return_10']):
            self.data['price_acceleration'] = self.data['return_5'] - self.data['return_10']
            self.data['momentum_ratio'] = self.data['return_5'] / (self.data['return_10'] + 1e-10)
        
        # Volatility-normalized returns
        if all(col in self.data.columns for col in ['return_1', 'historical_vol_20']):
            self.data['vol_normalized_return'] = self.data['return_1'] / (self.data['historical_vol_20'] + 1e-10)
        
        # Range expansion/contraction
        if all(col in self.data.columns for col in [self.high_col, self.low_col]):
            daily_range = self.data[self.high_col] - self.data[self.low_col]
            avg_range = daily_range.rolling(window=20).mean()
            self.data['range_expansion'] = (daily_range > avg_range * 1.5).astype(int)
            self.data['range_contraction'] = (daily_range < avg_range * 0.5).astype(int)
        
        # Price vs moving average distance
        if 'sma_20' in self.data.columns:
            self.data['price_vs_sma_pct'] = (
                (self.data[self.close_col] - self.data['sma_20']) / self.data['sma_20'] * 100
            )
                
        return self.data
    
    def get_all_custom_features(self,
                              returns_periods: List[int] = [1, 5, 10, 20],
                              volatility_periods: List[int] = [5, 10, 20, 50]):
        
        """
        Adds all custom features
        
        Parameters:
        returns_periods (List[int]): List of periods for returns calculation
        volatility_periods (List[int]): List of periods for volatility calculation
        
        """
        
        self.add_returns_features(returns_periods)
        self.add_volatility_measures(volatility_periods)
        self.add_price_position_features()
        self.add_seasonality_features()
        self.add_time_based_features()
        self.add_custom_derived_features()
        
        return self.data