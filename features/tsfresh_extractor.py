import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexTSFeatures:
    def __init__(
        self, 
        data: pd.DataFrame,
        target_column: str,
        instrument_id: str = "forex_pair",
    ):
        
        """
        Class for Automated Feature Engineering with TSFresh
        
        Parameters:
        data (pd.DataFrame): DataFrame with datetime index containing OHLCV + indicators
        target_column (str): Column name for target variable
        instrument_id (str): Identifier for the instrument (default: 'forex_pair')
        
        """
        
        print("="*50)
        print("TSFRESH FEATURE ENGINEERING")
        print("="*50)
        print(" Available Functions: \n1 prepare_ts_data \n2 extract_ts_features \n3 select_relevant_features \n4 get_feature_importance \n5 get_train_test_split \n6 get_all_features")
        print("="*50)
        
        self.data = data.copy()
        self.target_column = target_column
        self.instrument_id = instrument_id
        
        # Validate data
        self._validate_data()
        
        # Initialize results
        self.long_format_data = None
        self.extracted_features = None
        self.selected_features = None
        self.feature_importance = None
        self.scaler = None
        
    def _validate_data(self):
        
        """
        Validate input data
        
        """
        
        if self.data.index.name is None:
            raise ValueError("DataFrame index must have a name (datetime)")
        
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame")
        
        if self.data.empty:
            raise ValueError("DataFrame is empty")
    
    def prepare_ts_data(
        self, 
        feature_columns: Optional[List[str]] = None,
        window_size: int = 50,
        min_timeshift: int = 10,
    ):
        
        """
        Convert DataFrame to long format for tsfresh
        
        Parameters:
        feature_columns (List[str]): Columns to use for feature extraction
        window_size (int): Rolling window size
        min_timeshift (int): Minimum timeshift for rolling windows
        
        """
                
        # Use all numeric columns except target if not specified
        if feature_columns is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_cols if col != self.target_column]
                
        # Reset index to have datetime as column
        data_temp = self.data.reset_index()
        datetime_col = data_temp.columns[0]  # Get the name of the first column (datetime index)
        
        # Melt DataFrame to long format
        data_long = data_temp[[datetime_col] + feature_columns].melt(
            id_vars = [datetime_col],
            value_vars = feature_columns,
            var_name = "feature_type",
            value_name = "value"
        )
        
        # Add instrument ID
        data_long["id"] = self.instrument_id
        
        # Create rolling windows
        data_rolled = roll_time_series(
            data_long, 
            column_id = "id",
            column_sort = datetime_col,
            column_kind = "feature_type",
            max_timeshift = window_size,
            min_timeshift = min_timeshift
        )
        
        self.long_format_data = data_rolled
        self.datetime_column = datetime_col  # Store for later use
        
        return self.long_format_data
    
    def extract_ts_features(
        self, 
        n_jobs: int = 1,
        feature_settings: Optional[Dict] = None,
        disable_progressbar: bool = False
    ):

        """
        Extract features using tsfresh
        
        Parameters:
        n_jobs (int): Number of parallel jobs
        feature_settings (Dict): Custom feature extraction settings
        disable_progressbar (bool): Disable progress bar
        
        """
        
        if self.long_format_data is None:
            raise ValueError("Call prepare_ts_data() first")
                
        # Use comprehensive feature settings if not provided
        if feature_settings is None:
            feature_settings = ComprehensiveFCParameters()
            print("Using comprehensive feature settings")
        
        self.extracted_features = extract_features(
            self.long_format_data,
            column_id = "id",
            column_sort = self.datetime_column,
            column_kind = "feature_type",
            column_value = "value",
            default_fc_parameters = feature_settings,
            n_jobs = n_jobs,
            disable_progressbar = disable_progressbar
        )
        
        # Clean feature names
        self.extracted_features.columns = [f"tsfresh_{col}" for col in self.extracted_features.columns]
        
        self.extracted_features = self.extracted_features.replace([np.inf, -np.inf], np.nan)
        
        # Remove columns with too many NaN values
        initial_features = self.extracted_features.shape[1]
        self.extracted_features = self.extracted_features.dropna(axis=1, thresh=0.7 * len(self.extracted_features))
        self.extracted_features = self.extracted_features.ffill().bfill()
        
        final_features = self.extracted_features.shape[1]
        
        return self.extracted_features
    
    def select_relevant_features(
        self, 
        fdr_level: float = 0.05, 
        ml_task: str = 'regression',
    ):
        
        """
        Select statistically relevant features
        
        Parameters:
        fdr_level (float): FDR level for feature selection
        ml_task (str): Machine learning task (regression or classification)
        
        """
        
        if self.extracted_features is None:
            raise ValueError("Call extract_ts_features() first")
        
        print("Selecting relevant features...")
        
        # FIX: Get the unique time points from the rolled data to align with target
        unique_times = self.long_format_data[self.datetime_column].unique()
        
        # Create a mapping from time points to the extracted features index
        time_to_index_map = {}
        for idx, time_val in enumerate(unique_times):
            time_to_index_map[time_val] = idx
        
        # Align features with target using the time points - FIXED VERSION
        aligned_pairs = []
        
        for time_val in unique_times:
            if time_val in self.data.index:  # Check if time exists in original data
                if time_val in time_to_index_map:  # Check if time exists in mapping
                    feature_idx = time_to_index_map[time_val]
                    if feature_idx < len(self.extracted_features):  # Check if feature index is valid
                        target_value = self.data.loc[time_val, self.target_column]
                        aligned_pairs.append((time_val, feature_idx, target_value))
        
        if len(aligned_pairs) == 0:
            raise ValueError("No common time points between features and target")
        
        # Separate the aligned data
        aligned_indices = [pair[0] for pair in aligned_pairs]  # Time indices
        feature_indices = [pair[1] for pair in aligned_pairs]  # Feature indices  
        aligned_targets = [pair[2] for pair in aligned_pairs]  # Target values
        
        X_aligned = self.extracted_features.iloc[feature_indices]
        
        # FIX: Set the same index for both X and y
        X_aligned.index = aligned_indices  # Set index to match y
        y_aligned = pd.Series(aligned_targets, index=aligned_indices)
        
        print(f"Aligned data: {len(X_aligned)} samples")
        print(f"Target range: {y_aligned.min():.4f} to {y_aligned.max():.4f}")
        
        # Verify indices match
        print(f"X index: {X_aligned.index[:5].tolist()}")
        print(f"y index: {y_aligned.index[:5].tolist()}")
        print(f"Indices match: {X_aligned.index.equals(y_aligned.index)}")
        
        self.selected_features = select_features(
            X=X_aligned,
            y=y_aligned,
            fdr_level=fdr_level,
            ml_task=ml_task
        )
        
        # Store the aligned indices for later use
        self.aligned_indices = aligned_indices
        self.y_aligned = y_aligned
        
        print(f"Selected {self.selected_features.shape[1]} relevant features")
        
        return self.selected_features
    
    def get_feature_importance(
        self,
        top_n: int = 20
    ):
        
        """
        Calculate feature importance using correlation
        
        Parameters:
        top_n (int): Number of top features to return
        
        """
        
        if self.selected_features is None:
            raise ValueError("Call select_relevant_features() first")
        
        # Align with target using index
        common_index = self.selected_features.index.intersection(self.data.index)
        X_aligned = self.selected_features.loc[common_index]
        y_aligned = self.data.loc[common_index, self.target_column]
        
        # Calculate correlation with target
        correlations = X_aligned.apply(lambda x: np.corrcoef(x, y_aligned)[0, 1])
        correlations = correlations.abs().sort_values(ascending=False)
        
        self.feature_importance = correlations.head(top_n)
        
        print(" Top feature correlations:")
        for i, (feature, corr) in enumerate(self.feature_importance.items(), 1):
            actual_corr = np.corrcoef(X_aligned[feature], y_aligned)[0, 1]
            direction = "+" if actual_corr > 0 else "-"
            print(f"  {i:2d}. {feature}: {corr:.4f} ({direction})")
        
        return self.feature_importance
        
    def get_train_test_split(
        self, 
        test_size: float = 0.2,
        scale_features: bool = True,
    ):

        """
        Prepare train/test splits (time-series aware)
        
        Parameters:
        test_size (float): Proportion of test set
        scale_features (bool): Whether to scale features
        
        """
        
        if self.selected_features is None:
            raise ValueError("Call select_relevant_features() first")
                
        # Align features and target using index
        common_index = self.selected_features.index.intersection(self.data.index)
        X = self.selected_features.loc[common_index]
        y = self.data.loc[common_index, self.target_column]
        
        # Time-series split without shuffling - preserve time order
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert back to DataFrame with original index and columns
            X_train = pd.DataFrame(X_train_scaled, 
                                 index=X_train.index, 
                                 columns=X_train.columns)
            X_test = pd.DataFrame(X_test_scaled, 
                                index=X_test.index, 
                                columns=X_test.columns)
            self.scaler = scaler
            print("Features scaled using StandardScaler")
                
        return X_train, X_test, y_train, y_test
    
    def get_all_features(
        self,
        feature_columns: Optional[List[str]] = None,
        window_size: int = 50,
        test_size: float = 0.2,
        scale_features: bool = True,
    ):
        
        """
        Complete pipeline: preparation → extraction → selection → split
        
        Parameters:
        feature_columns (List[str]): Columns for feature extraction
        window_size (int): Rolling window size
        test_size (float): Test set proportion
        scale_features (bool): Whether to scale features
        
        """
        
        # Step 1: Prepare data
        self.prepare_ts_data(feature_columns=feature_columns, window_size=window_size)
        
        # Step 2: Extract features
        self.extract_ts_features()
        
        # Step 3: Select relevant features
        self.select_relevant_features()
        
        # Step 4: Get feature importance
        self.get_feature_importance()
        
        # Step 5: Train/test split
        X_train, X_test, y_train, y_test = self.get_train_test_split(
            test_size=test_size, 
            scale_features=scale_features
        )
        
        return X_train, X_test, y_train, y_test

