import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

class ForexDataClean:
    def __init__(
        self, 
        data: pd.DataFrame, 
        columns: list = None,
        open_col: str = 'open',
        high_col: str = 'high', 
        low_col: str = 'low', 
        close_col: str = 'close',
        volume_col: str = 'volume',
        prints: bool = True,
    ):
        
        """
        Initialize the Forex Data Cleaner
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data
        columns (list): List of columns to clean
        fast_clean (bool): Whether to perform fast cleaning upon initialization
        
        """
        
        self.prints = prints
        
        if self.prints:
            print("="*50)
            print("FOREX DATA CLEANER")
            print("="*50)
            print(" Available Fuctions \n1 remove_duplicates \n2 handle_missing_values \n3 validate_ohlc_integrity \n4 handle_outliers \n5 fast_cleaner")
            print("="*50)
    
        self.data = data.copy()
        self.original_data = data.copy()
        
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        
        self.columns = columns if columns is not None else data.columns
        
    def reset_data(
        self
    ):
        
        """
        Reset the data to its original state
        
        """
        
        self.data = self.original_data.copy()
        
        return self.data
        
    def remove_duplicates(
        self, 
        subset: list = None, 
        keep: str = 'first'
    ):
        
        """
        Remove duplicate entries
        
        Parameters:
        subset: Columns to check for duplicates
        keep: Which duplicate to keep ('first', 'last', False)
        
        """
        before_drop = len(self.data)
        
        if subset is None:
            self.data = self.data.drop_duplicates(keep = keep)
        else:
            if subset == self.data.index.name:
                temp_data = self.data.reset_index()
                temp_data = temp_data.drop_duplicates(subset = subset, keep = keep)
                self.data = temp_data.set_index(subset)
            else:
                self.data = self.data.drop_duplicates(subset = subset, keep = keep) 
        after_drop = len(self.data)

        if self.prints:
            print("="*50)
            print(f"Keep = {keep} and subset = {subset}")
            print(f"Removed {before_drop - after_drop} duplicate entries")
        
        self._validate_ohlc_integrity()
        
        return self.data
            
    def handle_missing_values(
        self, 
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
        
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(self.data[col])]
        
        missing_values = self.data[columns].isnull().sum().sum()

        if missing_values == 0:
            print("No missing values found")
        else:
            print(f"Found {missing_values} missing values")        
            if method == 'interpolate':
                self.data[numeric_cols] = self.data[numeric_cols].interpolate(method='linear')
                affected = self.data[columns].isna().sum().sum()
                print(f" interpolated: {missing_values - affected} missing values .")
            elif method == 'ffill':
                self.data[numeric_cols] = self.data[numeric_cols].fillna(method='ffill')
                affected = self.data[columns].isna().sum().sum()
                print(f" forward filled: {missing_values - affected} missing values .")
            elif method == 'bfill':
                self.data[numeric_cols] = self.data[numeric_cols].fillna(method='bfill')
                affected = self.data[columns].isna().sum().sum()
                print(f" backward filled: {missing_values - affected} missing values .")
            elif method == 'drop':
                self.data = self.data.dropna(subset=columns)
                affected = self.data[columns].isna().sum().sum()
                print(f" dropped: {missing_values - affected} missing values .")
            elif method == 'zero':
                self.data[numeric_cols] = self.data[numeric_cols].fillna(0)
                affected = self.data[columns].isna().sum().sum()
                print(f" filled with zeros: {missing_values - affected} missing values .")
            else:
                print("Invalid method")
            
        remaining_missing = self.data[columns].isna().sum().sum()
        if remaining_missing > 0:
            print(f"Still have {remaining_missing} missing values.")
    
        return self.data    
            
    def _validate_ohlc_integrity(
        self, 
    ):
        
        """
        Validate OHLC data integrity and optionally fix errors
        
        """
        
        high_low_violations = (self.data[self.high_col] < self.data[self.low_col])
        open_low_violations = (self.data[self.open_col] < self.data[self.low_col])
        open_high_violations = (self.data[self.open_col] > self.data[self.high_col])
        close_low_violations = (self.data[self.close_col] < self.data[self.low_col])
        close_high_violations = (self.data[self.close_col] > self.data[self.high_col])
        
        open_violations = open_low_violations | open_high_violations
        close_violations = close_low_violations | close_high_violations
        
        violation_count = high_low_violations.sum() +  open_violations.sum() +  close_violations.sum()    
        if self.prints:
            print("=" * 50)
            print("OHLC DATA INTEGRITY VALIDATION")
            print("=" * 50)
        
        if violation_count == 0:
            print("No OHLC integrity violations found")
            print("All OHLC values are consistent")
            return violation_count
        
        if self.prints:
            print(f"Found {violation_count} OHLC integrity violations:")
            print(f"-High < Low violations: {high_low_violations.sum()}")
            print(f"-Open price violations: {open_violations.sum()}")
            print(f"     • Open < Low: {open_low_violations.sum()}")
            print(f"     • Open > High: {open_high_violations.sum()}")
            print(f"- Close price violations: {close_violations.sum()}")
            print(f"     • Close < Low: {close_low_violations.sum()}")
            print(f"     • Close > High: {close_high_violations.sum()}")
                                 
        if violation_count > 0:
            print(f"Found {violation_count} OHLC integrity violations")
    
    def handle_outliers(
        self, 
        method: str =' quantile', 
        threshold: float = 3.0, 
        strategy: str = 'cap', 
        columns: list = None,
        remove: bool = True
    ):
        
        """
        Handle outliers in numeric columns
        
        Parameters:
        method: 'iqr', 'zscore', 'quantile'
        threshold: Threshold for outlier detection
        strategy: 'remove', 'cap', 'mean', 'median'
        columns: Specific columns to process
        remove: Whether to actually remove outliers or just detect them
        
        """
        
        if columns is None:
            columns = self.columns
            
        if self.prints:
            print("=" * 50)
            print("OUTLIER DETECTION AND HANDLING")
            print("=" * 50)
            print(f"Method: {method}, Strategy: {strategy}, Threshold: {threshold}")
            print(f"Remove outliers: {remove}")
            print("-" * 50)
        
        total_outliers = 0
        total_rows_before = len(self.data)
        
        for col in columns:
            
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                continue
            
            original_min = self.data[col].min()
            original_max = self.data[col].max()
            original_mean = self.data[col].mean()
            original_std = self.data[col].std()
            
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                method_desc = f"IQR (Q1={Q1:.5f}, Q3={Q3:.5f}, IQR={IQR:.5f})"
                
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                lower_bound = self.data[col].mean() - threshold * self.data[col].std()
                upper_bound = self.data[col].mean() + threshold * self.data[col].std()
                method_desc = f"Z-score (μ={self.data[col].mean():.5f}, σ={self.data[col].std():.5f})"
                
            elif method == 'quantile':
                lower_bound = self.data[col].quantile(0.01)
                upper_bound = self.data[col].quantile(0.99)
                method_desc = f"Quantile (1%-99%)"
            
            lower_outliers = (self.data[col] < lower_bound).sum()
            upper_outliers = (self.data[col] > upper_bound).sum()
            col_outliers = lower_outliers + upper_outliers
            
            if col_outliers == 0:
                print(f"{col}: No outliers found [{original_min:.5f} - {original_max:.5f}]")
                continue
            
            total_outliers += col_outliers
            
            if self.prints:
                print(f"{col}: Found {col_outliers} outliers ({lower_outliers} low, {upper_outliers} high)")
                print(f"Bounds: [{lower_bound:.5f} - {upper_bound:.5f}]")
                print(f"Original range: [{original_min:.5f} - {original_max:.5f}]")
                print(f"Method: {method_desc}")
            
            if remove:
                if strategy == 'remove':
                    rows_before = len(self.data)
                    self.data = self.data[(self.data[col] >= lower_bound) & 
                                          (self.data[col] <= upper_bound)]
                    
                    rows_removed = rows_before - len(self.data)
                    print(f"Removed {rows_removed} rows containing outliers")
                    
                elif strategy == 'cap':
                    capped_low = (self.data[col] < lower_bound).sum()
                    capped_high = (self.data[col] > upper_bound).sum()
                    self.data[col] = np.clip(self.data[col], lower_bound, upper_bound)
                    print(f" Capped {capped_low} values to lower bound, {capped_high} to upper bound")
                    
                elif strategy == 'mean':
                    mean_val = self.data[col].mean()
                    low_replace = (self.data[col] < lower_bound).sum()
                    high_replace = (self.data[col] > upper_bound).sum()
                    self.data.loc[self.data[col] < lower_bound, col] = mean_val
                    self.data.loc[self.data[col] > upper_bound, col] = mean_val
                    print(f"Replaced {low_replace + high_replace} outliers with mean ({mean_val:.5f})")
                    
                elif strategy == 'median':
                    median_val = self.data[col].median()
                    low_replace = (self.data[col] < lower_bound).sum()
                    high_replace = (self.data[col] > upper_bound).sum()
                    self.data.loc[self.data[col] < lower_bound, col] = median_val
                    self.data.loc[self.data[col] > upper_bound, col] = median_val
                    print(f"Replaced {low_replace + high_replace} outliers with median ({median_val:.5f})")
                
                new_min = self.data[col].min()
                new_max = self.data[col].max()
                print(f"New range: [{new_min:.5f} - {new_max:.5f}]")
            else:
                print(f"Outliers detected but not removed (remove=False)")
            
            print()
        
        if self.prints:
            print("=" * 50)
            if total_outliers == 0:
                print("No outliers found in any numeric columns!")
            else:
                rows_after = len(self.data)
                rows_removed_total = total_rows_before - rows_after
                
                if remove:
                    if strategy == 'remove':
                        print(f"SUMMARY: Removed {rows_removed_total} rows containing {total_outliers} outliers")
                    else:
                        print(f"SUMMARY: Processed {total_outliers} outliers across all columns")
                        print(f"   Rows before: {total_rows_before}, Rows after: {rows_after}")
                else:
                    print(f"SUMMARY: Found {total_outliers} outliers (not removed)")
                    print(f"Use remove=True to handle these outliers")
            
            print("=" * 50)
            
        return self.data
        
    def fast_cleaner(
        self,
        missing_method: str = 'drop', 
        missing_columns: list = None,
        duplicates_columns: list = None, 
        duplicates_method: str='first',
        outlier_method: str = 'quantile', 
        outlier_threshold: float=3.0, 
        outlier_strategy: str ='cap', 
        outlier_columns: list = None,
        Handle_outliers: bool = False,
        prints: bool = False
    ):
        
        """
        Fast clean 
        
        Parameters:
        missing_method: 'drop', 'interpolate', 'ffill', 'bfill', 'zero'
        missing_columns: Specific columns to handle (None for all OHLCV)
        duplicates_columns: Columns to check for duplicates
        duplicates_method: 'first', 'last', False
        outlier_method: 'iqr', 'zscore', 'quantile'
        outlier_threshold: Threshold for outlier detection
        outlier_strategy: 'remove', 'cap', 'mean', 'median'
        outlier_columns: Specific columns to process
        Handle_outliers: Whether to actually handle outliers or just detect them
         
        """
        
        self.remove_duplicates(
            subset = duplicates_columns, 
            keep = duplicates_method
        )
        if Handle_outliers:
            self.handle_outliers(
                method = outlier_method, 
                threshold = outlier_threshold, 
                strategy = outlier_strategy, 
                columns = outlier_columns
            )
        self.handle_missing_values(
            method = missing_method, 
            columns = missing_columns
        )
        self._validate_ohlc_integrity()
        
        return self.data


