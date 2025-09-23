import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

#ADD PRINTS AND EXPLANATIONS 

class ForexDataClean:
    def __init__(
        self, 
        data: pd.DataFrame, 
        columns: list = None,
        fast_clean: bool = False, 
    ):
        
        """
        Initialize the Forex Data Cleaner
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data
        columns (list): List of columns to clean
        fast_clean (bool): Whether to perform fast cleaning upon initialization
        
        """
        print("="*50)
        print("FOREX DATA CLEANER")
        print("="*50)
        print(" Available Fuctions \n1 remove_duplicates \n2 handle_missing_values \n3 validate_ohlc_integrity \n4 handle_outliers")
        print("="*50)
    
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
        
    def remove_duplicates(
        self, 
        subset: list = None, 
        keep='first'
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

        print("="*50)
        print(f"Keep = {keep} and subset = {subset}")
        print(f"Removed {before_drop - after_drop} duplicate entries")
        
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
            
        # Count Missing values
        missing_values = self.data[columns].isnull().sum().sum()

        # Working with the Missing values with the sellected method
        if missing_values == 0:
            print("No missing values found")
        else:
            print(f"Found {missing_values} missing values")        
            if method == 'interpolate':
                self.data[columns] = self.data[columns].interpolate(method='linear')
                affected = self.data[columns].isna().sum().sum()
                print(f" interpolated: {missing_values - affected} missing values .")
            elif method == 'ffill':
                self.data[columns] = self.data[columns].fillna(method='ffill')
                affected = self.data[columns].isna().sum().sum()
                print(f" forward filled: {missing_values - affected} missing values .")
            elif method == 'bfill':
                self.data[columns] = self.data[columns].fillna(method='bfill')
                affected = self.data[columns].isna().sum().sum()
                print(f" backward filled: {missing_values - affected} missing values .")
            elif method == 'drop':
                self.data = self.data.dropna(subset=columns)
                affected = self.data[columns].isna().sum().sum()
                print(f" dropped: {missing_values - affected} missing values .")
            elif method == 'zero' and self.volume_col in columns:
                self.data[self.volume_col] = self.data[self.volume_col].fillna(0)
                affected = self.data[columns].isna().sum().sum()
                print(f" filled with zeros: {missing_values - affected} missing values .")
            else:
                print("Invalid method")
            
        remaining_missing = self.data[columns].isna().sum().sum()
        if remaining_missing > 0:
            print(f"Still have {remaining_missing} missing values.")
    
        return self.data    
            
    def validate_ohlc_integrity(
        self, 
        fix_errors = False
    ):
        
        """
        Validate OHLC data integrity and optionally fix errors
        
        Parameters:
        fix_errors: Whether to automatically fix detected errors
        
        """
        
        # Check for various types of OHLC violations
        high_low_violations = (self.data['high'] < self.data['low'])
        open_low_violations = (self.data['open'] < self.data['low'])
        open_high_violations = (self.data['open'] > self.data['high'])
        close_low_violations = (self.data['close'] < self.data['low'])
        close_high_violations = (self.data['close'] > self.data['high'])
        
        open_violations = open_low_violations | open_high_violations
        close_violations = close_low_violations | close_high_violations
        
        violation_count = high_low_violations.sum() +  open_violations.sum() +  close_violations.sum()    
        
        print("=" * 50)
        print("OHLC DATA INTEGRITY VALIDATION")
        print("=" * 50)
        
        if violation_count == 0:
            print("No OHLC integrity violations found")
            print("All OHLC values are consistent")
            return violation_count
        
        # Detailed violation report

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
            
        #if fix_errors:
        #     print(f"\n Attempting to fix {violation_count} violations...")
            
        #     # Fix High < Low violations by swapping
        #     hl_fix_count = high_low_violations.sum()
        #     if hl_fix_count > 0:
        #         self.data.loc[high_low_violations, ['high', 'low']] = self.data.loc[high_low_violations, ['low', 'high']].values
        #         print(f"Fixed {hl_fix_count} High < Low violations by swapping values")
            
        #     # Fix Open price violations by clamping to [low, high] range
        #     open_fix_count = open_violations.sum()
        #     if open_fix_count > 0:
        #         self.data.loc[open_low_violations, 'open'] = self.data.loc[open_low_violations, 'low']
        #         self.data.loc[open_high_violations, 'open'] = self.data.loc[open_high_violations, 'high']
        #         print(f"Fixed {open_fix_count} Open price violations by clamping to valid range")
            
        #     # Fix Close price violations by clamping to [low, high] range
        #     close_fix_count = close_violations.sum()
        #     if close_fix_count > 0:
        #         self.data.loc[close_low_violations, 'close'] = self.data.loc[close_low_violations, 'low']
        #         self.data.loc[close_high_violations, 'close'] = self.data.loc[close_high_violations, 'high']
        #         print(f"Fixed {close_fix_count} Close price violations by clamping to valid range")
            
        #     # Verify fixes
        #     remaining_violations = self.validate_ohlc_integrity(fix_errors=False)
        #     if remaining_violations == 0:
        #         print(" All violations successfully fixed!")
        #     else:
        #         print(f"Could not fix all violations. {remaining_violations} remain.")
            
        #     return remaining_violations
        # else:
        #     print(f"\n Use fix_errors=True to automatically fix these violations")
        
        # return violation_count
    
    def handle_outliers(
        self, 
        method='quantile', 
        threshold=3.0, 
        strategy='cap', 
        columns: list = None,
        remove: bool = False
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
        
        print("=" * 50)
        print("OUTLIER DETECTION AND HANDLING")
        print("=" * 50)
        print(f"Method: {method}, Strategy: {strategy}, Threshold: {threshold}")
        print(f"Remove outliers: {remove}")
        print("-" * 50)
        
        total_outliers = 0
        total_rows_before = len(self.data)
        
        for col in columns:
            
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(self.data[col]):

                print(f"⏭️  Skipping non-numeric column: {col}")
                continue
            
            # Store original stats
            original_min = self.data[col].min()
            original_max = self.data[col].max()
            original_mean = self.data[col].mean()
            original_std = self.data[col].std()
            
            # Calculate bounds based on method
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
            
            # Identify outliers
            lower_outliers = (self.data[col] < lower_bound).sum()
            upper_outliers = (self.data[col] > upper_bound).sum()
            col_outliers = lower_outliers + upper_outliers
            
            if col_outliers == 0:
                print(f"{col}: No outliers found [{original_min:.5f} - {original_max:.5f}]")
                continue
            
            total_outliers += col_outliers
            print(f"{col}: Found {col_outliers} outliers ({lower_outliers} low, {upper_outliers} high)")
            print(f"Bounds: [{lower_bound:.5f} - {upper_bound:.5f}]")
            print(f"Original range: [{original_min:.5f} - {original_max:.5f}]")
            print(f"Method: {method_desc}")
            
            if remove:
                # Apply outlier handling strategy
                if strategy == 'remove':
                    rows_before = len(self.data)
                    self.data = self.data[(self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)]
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
                
                # Show after stats
                new_min = self.data[col].min()
                new_max = self.data[col].max()
                print(f"New range: [{new_min:.5f} - {new_max:.5f}]")
            else:
                print(f"Outliers detected but not removed (remove=False)")
            
            print()
        
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
        
    def fast_cleaner(self):
        
        """
        Fast clean 
         
        """
        
        self.remove_duplicates()
        self.handle_missing_values()
        self.handle_outliers()
        self.validate_ohlc_integrity()
        
        return self.data


