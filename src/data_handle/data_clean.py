import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

class ForexDataClean:
    def __init__(self, 
                 data: pd.DataFrame, 
                 columns: list = None,
                 fast_clean: bool = False, 
                 ):
    
        self.data = data.copy()
        self.original_data = data.copy()
        
        if columns is not None:
            self.columns = columns
        else:
            self.columns = self.data.columns
            
        if fast_clean:
            self.fast_cleaner()
        
    def reset_data(self):
        
        """
        Reset the data to its original state
        
        """
        
        self.data = self.original_data.copy()
        
        return self.data
        
    def remove_duplicates(self, 
                          subset: list = None, 
                          keep='first'
                          ):
        
        """
        Remove duplicate entries
        
        Parameters:
        subset: Columns to check for duplicates
        keep: Which duplicate to keep ('first', 'last', False)
        
        """
        
        if subset is None:
            subset = self.data.index.name
        
        self.data = self.data.drop_duplicates(subset = subset, keep = keep)
        
        return self.data
            
    def handle_missing_values(self, 
                              method: str = 'drop', 
                              columns: list = None
                              ):
        
        """
        Handle missing values in specified columns
        
        Parameters:
        method: 'interpolate', 'ffill', 'bfill', 'drop', 'zero'
        columns: Specific columns to handle (None for all OHLCV)
        
        """
        
        if columns is None:
            columns = self.columns
        
        if method == 'interpolate':
            self.data[columns] = self.data[columns].interpolate(method='linear')
        elif method == 'ffill':
            self.data[columns] = self.data[columns].fillna(method='ffill')
        elif method == 'bfill':
            self.data[columns] = self.data[columns].fillna(method='bfill')
        elif method == 'drop':
            self.data = self.data.dropna(subset=columns)
        elif method == 'zero' and self.volume_col in columns:
            self.data[self.volume_col] = self.data[self.volume_col].fillna(0)
            
        return self.data    
            
    def validate_ohlc_integrity(self, fix_errors = False):
        
        """
        Validate OHLC data integrity and optionally fix errors
        
        Parameters:
        fix_errors: Whether to automatically fix detected errors
        
        """
        
        # Check High >= Low
        high_low_violations = (self.data['high'] < self.data['low'])
        
        # Check Open/Close within range
        open_violations = ((self.data['open'] < self.data['low']) |
                           (self.data['open'] > self.data['high']))
        
        close_violations = ((self.data['close'] < self.data['low']) |
                            (self.data['close'] > self.data['high']))         
                             
        violation_count = high_low_violations.sum() + open_violations.sum() + close_violations.sum()
    
        if violation_count > 0:
            print(f"Found {violation_count} OHLC integrity violations")
            
            if fix_errors:
                pass
        
        return violation_count
    
    def handle_outliers(self, 
                        method='quantile', 
                        threshold=3.0, 
                        strategy='cap', 
                        columns: list = None
                        ):
        
        """
        Handle outliers in numeric columns
        
        Parameters:
        method: 'iqr', 'zscore', 'quantile'
        threshold: Threshold for outlier detection
        strategy: 'remove', 'cap', 'mean', 'median'
        columns: Specific columns to process
        
        """
        
        if columns is None:
            columns = self.columns
        
        for col in columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                lower_bound = self.data[col].mean() - threshold * self.data[col].std()
                upper_bound = self.data[col].mean() + threshold * self.data[col].std()
            
            elif method == 'quantile':
                lower_bound = self.data[col].quantile(0.01)
                upper_bound = self.data[col].quantile(0.99)
            
            if strategy == 'remove':
                self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
            elif strategy == 'cap':
                self.data[col] = np.clip(self.data[col], lower_bound, upper_bound)
            elif strategy == 'mean':
                mean_val = self.data[col].mean()
                self.data.loc[self.data[col] < lower_bound, col] = mean_val
                self.data.loc[self.data[col] > upper_bound, col] = mean_val
            elif strategy == 'median':
                median_val = self.data[col].median()
                self.data.loc[self.data[col] < lower_bound, col] = median_val
                self.data.loc[self.data[col] > upper_bound, col] = median_val
        
        return self.data
    
    def fast_cleaner(self):
        
        self.remove_duplicates()
        self.handle_missing_values()
        self.validate_ohlc_integrity()
        
        return self.data
    

