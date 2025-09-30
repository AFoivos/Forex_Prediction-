import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexCustomIndicators:
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
        print("CUSTOM INDICATORS")
        print("="*50)
        print(" Available Fuctions \n1 add_returns_indicators \n2 add_volatility_measures_indicators \n3 add_price_position_indicators \n4 add_seasonality_indicators \n5 add_time_based_indicators \n6 add_custom_derived_indicators \n7 generate_all_custom_indicators")
        print("="*50)
        
        self.data = data.copy()
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        
        self.parameters = {}
        
        self.custom_data = pd.DataFrame(
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
        
        if not all(isinstance(item, list) for item in lst):
            return [lst]
        else:
            return lst
    
    def add_returns_indicators(
        self, 
        periods: List[int] = [1, 4, 24, 120],
    ):
        
        """
        Returns and Logarithmic Returns indicators
        
        Parameters:
        periods (List[int]): List of periods for returns calculation
        
        """
        
        self.parameters['ret_params'] = periods
                
        for period in periods:
            # Simple returns
            ret_col = f'ret_{period}'
            self.custom_data[ret_col] = self.data[self.close_col].pct_change(period)
            
            # Logarithmic returns
            log_ret_col = f'log_ret_{period}'
            self.custom_data[log_ret_col] = np.log(self.data[self.close_col] / self.data[self.close_col].shift(period))
            
            # Returns volatility (rolling std)
            vol_col = f'vol_{period}'
            self.custom_data[vol_col] = self.custom_data[ret_col].rolling(window = period).std()
            
            # Returns direction
            dir_col = f'ret_direction_{period}'
            self.custom_data[dir_col] = np.where(
                self.custom_data[ret_col] > 0, 
                1,
                -1
            )
                    
            # Cumulative returns
            self.custom_data[f'cumulative_return_{period}'] = (1 + self.custom_data[ret_col]).cumprod() - 1
            self.custom_data[f'cumulative_log_return_{period}'] = self.custom_data[log_ret_col].cumsum()
        
        return self.custom_data
    
    def add_volatility_measures_indicators(
        self,
        periods: List[int] = [5, 10, 20, 50],
    ):
        
        """
        Advanced Volatility Measures indicators
        
        Parameters:
        periods (List[int]): List of periods for volatility calculation
        
        """
        
        self.parameters['volatility_mesures_params'] = periods
        
        if 'log_ret_1' not in self.custom_data.columns:
            self.add_returns_indicators(periods=[1])
        
        for period in periods:
            # Historical volatility (annualized)
            hv_col = f'historical_vol_{period}'
            daily_vol = self.custom_data['log_ret_1'].rolling(window=period).std()
            self.custom_data[hv_col] = daily_vol * np.sqrt(252)  
            
            # Parkinson volatility (high-low based)
            parkinson_col = f'parkinson_vol_{period}'
            log_hl = np.log(self.data[self.high_col] / self.data[self.low_col])
            self.custom_data[parkinson_col] = np.sqrt((1/(4 * period * np.log(2))) * (log_hl**2).rolling(window=period).sum())
            
            # Garman-Klass volatility
            gk_col = f'garman_klass_vol_{period}'
            log_hl_sq = (np.log(self.data[self.high_col] / self.data[self.low_col]))**2
            log_co_sq = (2 * np.log(2) - 1) * (np.log(self.data[self.close_col] / self.data[self.open_col]))**2
            self.custom_data[gk_col] = np.sqrt((1/period) * (log_hl_sq - log_co_sq).rolling(window=period).sum())
            
            # Volatility ratio (close-to-close vs high-low)
            vol_ratio_col = f'vol_ratio_{period}'
            self.custom_data[vol_ratio_col] = self.custom_data[parkinson_col] / (self.custom_data[hv_col] + 1e-10)
                    
        # Volatility regimes
        self.custom_data['high_volatility_regime'] = (self.custom_data['historical_vol_20'] > self.custom_data['historical_vol_20'].quantile(0.75)).astype(int)
        self.custom_data['low_volatility_regime'] = (self.custom_data['historical_vol_20'] < self.custom_data['historical_vol_20'].quantile(0.25)).astype(int)
        
        return self.custom_data
    
    def add_price_position_indicators(self):
        
        """
        Price Position in Range indicators
        
        """
                
        # Daily price range position
        self.custom_data['daily_range_pct'] = (
            (self.data[self.close_col] - self.data[self.low_col]) / 
            (self.data[self.high_col] - self.data[self.low_col] + 1e-10)
        ) * 100
        
        # Weekly range position (5-day rolling)
        self.custom_data['weekly_high'] = self.data[self.high_col].rolling(window=5).max()
        self.custom_data['weekly_low'] = self.data[self.low_col].rolling(window=5).min()
        self.custom_data['weekly_range_pct'] = (
            (self.custom_data[self.close_col] - self.custom_data['weekly_low']) / 
            (self.custom_data['weekly_high'] - self.custom_data['weekly_low'] + 1e-10)
        ) * 100
        
        # Monthly range position (20-day rolling)
        self.custom_data['monthly_high'] = self.data[self.high_col].rolling(window=20).max()
        self.custom_data['monthly_low'] = self.data[self.low_col].rolling(window=20).min()
        self.custom_data['monthly_range_pct'] = (
            (self.custom_data[self.close_col] - self.custom_data['monthly_low']) / 
            (self.custom_data['monthly_high'] - self.custom_data['monthly_low'] + 1e-10)
        ) * 100
        
        # Range position signals
        self.custom_data['at_daily_high'] = (self.data[self.close_col] == self.data[self.high_col]).astype(int)
        self.custom_data['at_daily_low'] = (self.data[self.close_col] == self.data[self.low_col]).astype(int)
        
        self.custom_data['near_weekly_high'] = (self.custom_data['weekly_range_pct'] > 80).astype(int)
        self.custom_data['near_weekly_low'] = (self.custom_data['weekly_range_pct'] < 20).astype(int)
        
        return self.custom_data
    
    def add_seasonality_indicators(self):
        
        """
        Seasonality and Calendar indicators
        
        """
        
        seasonality_df = pd.DataFrame(index=self.data.index)
        
        # Time-based features
        seasonality_df['hour'] = self.data.index.hour
        seasonality_df['day_of_week'] = self.data.index.dayofweek
        seasonality_df['day_of_month'] = self.data.index.day
        seasonality_df['week_of_year'] = self.data.index.isocalendar().week.values  
        seasonality_df['month'] = self.data.index.month
        seasonality_df['quarter'] = self.data.index.quarter
        
        # Time of day segments
        seasonality_df['time_of_day'] = pd.cut(
            seasonality_df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        # Day type
        seasonality_df['is_weekend'] = (seasonality_df['day_of_week'] >= 5).astype(int)
        seasonality_df['is_month_end'] = (self.data.index.is_month_end).astype(int)
        seasonality_df['is_quarter_end'] = (self.data.index.is_quarter_end).astype(int)
        seasonality_df['is_year_end'] = (self.data.index.is_year_end).astype(int)
        
        # Seasonal indicators
        seasonality_df['season'] = pd.cut(
            seasonality_df['month'],
            bins=[0, 3, 6, 9, 12],
            labels=['winter', 'spring', 'summer', 'fall'],
            include_lowest=True
        )
        
        # Add to custom_data to keep index
        for col in seasonality_df.columns:
            self.custom_data[col] = seasonality_df[col]
        
        return self.custom_data
    
    def add_time_based_indicators(self):
    
        """
        Advanced Time-based indicators
        
        """
                
        time_df = pd.DataFrame(index=self.data.index)
        
        if 'day_of_week' not in self.custom_data.columns or 'hour' not in self.custom_data.columns:
            self.add_seasonality_indicators()
        
        # Time since market open (assuming 24/5 market)
        time_df['minutes_since_monday_open'] = (
            (self.custom_data['day_of_week'] * 24 * 60) + 
            (self.custom_data['hour'] * 60) + 
            self.data.index.minute
        )
        
        # Period of day (for intraday patterns)
        time_df['period_of_day'] = pd.cut(
            self.custom_data['hour'],
            bins=[0, 4, 8, 12, 16, 20, 24],
            labels=['late_night', 'early_morning', 'morning', 'afternoon', 'evening', 'night'],
            include_lowest=True
        )
        
        # Holiday proximity (simplified) - με προστασία index
        month_end = (self.data.index + pd.offsets.MonthEnd(0))
        quarter_end = (self.data.index + pd.offsets.QuarterEnd(0))
        
        time_df['days_to_month_end'] = (month_end - self.data.index).days
        time_df['days_to_quarter_end'] = (quarter_end - self.data.index).days
        
        # Add to custom_data to keep index
        for col in time_df.columns:
            self.custom_data[col] = time_df[col]
        
        return self.custom_data
    
    def add_custom_derived_indicators(self):
        
        """
        Custom Derived indicators
        
        """
                        
        # Price momentum features
        if 'ret_5' not in self.custom_data.columns:
            self.add_returns_indicators(periods = [5])
        if 'ret_10' not in self.custom_data.columns:
            self.add_returns_indicators(periods = [10])
            
        self.custom_data['price_acceleration'] = self.custom_data['ret_5'] - self.custom_data['ret_10']
        self.custom_data['momentum_ratio'] = self.custom_data['ret_5'] / (self.custom_data['ret_10'] + 1e-10)
        
        # Volatility-normalized returns
        if 'ret_1' not in self.custom_data.columns:
            self.add_returns_indicators(periods = [1])
        if 'historical_vol_20' not in self.custom_data.columns:
            self.add_volatility_measures_indicators(periods = [20])
        
        self.custom_data['vol_normalized_return'] = self.custom_data['ret_1'] / (self.custom_data['historical_vol_20'] + 1e-10)
        
        # Range expansion/contraction
        daily_range = self.data[self.high_col] - self.data[self.low_col]
        avg_range = daily_range.rolling(window=20).mean()
        self.custom_data['range_expansion'] = (daily_range > avg_range * 1.5).astype(int)
        self.custom_data['range_contraction'] = (daily_range < avg_range * 0.5).astype(int)
        
        # Price vs moving average distance
        if 'sma_20' not in self.data.columns:
            self.data['sma_20'] = talib.SMA(
                self.data[self.close_col], 
                timeperiod=20
            )
            
        self.custom_data['price_vs_sma_20_pct'] = (
            (self.data[self.close_col] - self.data['sma_20']) / self.data['sma_20'] * 100
        )
                
        return self.custom_data
    
    def generate_all_custom_indicators(
        self,
        returns_periods: List[int] = [1, 4, 24, 120],
        volatility_periods: List[int] = [5, 10, 20, 50],
    ):
        
        """
        Adds all custom features
        
        Parameters:
        returns_periods (List[int]): List of periods for returns calculation
        volatility_periods (List[int]): List of periods for volatility calculation
        
        """
        
        self.add_returns_indicators(periods = returns_periods)
        self.add_volatility_measures_indicators(periods = volatility_periods)
        self.add_price_position_indicators()
        self.add_seasonality_indicators()
        self.add_time_based_indicators()
        self.add_custom_derived_indicators()
        
        count_removed_rows = self.data.shape[0] - self.data.shape[0]
        
        print('='*50)
        print('Data Info')
        print(self.custom_data.info())
        print('='*50)
        print(f'Shape of data {self.custom_data.shape}')
        print('='*50)
        print(f'{count_removed_rows} rows removed')
        print('='*50)
        
        return self.custom_data, self.parameters