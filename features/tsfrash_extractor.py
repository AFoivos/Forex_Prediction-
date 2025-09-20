import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
import joblib

import warnings
warnings.filterwarnings('ignore')

class ForexFeatureExtractor:
    def __init__(self, 
                 data: pd.DataFrame, 
                 default_fc_parameters=None):
        
        """ 
        Class for feature extraction using TSfresh
        
        Parameters:
        data (pd.DataFrame): Input dataframe with OHLC data and indicators
        default_fc_parameters (dict): Default feature extraction parameters
        
        """
        
        print("="*50)
        print("FEATURE EXTRACTION")
        print("="*50)
        print("Available Functions: \n1. extract_features \n2. prepare_tsfresh_data \n3. select_relevant_features \n4. save_features \n5. load_features")
        
        if default_fc_parameters is None:
            self.fc_parameters = {
                # Basic Statistics
                'length': None,
                'mean': None,
                'median': None,
                'standard_deviation': None,
                'variance': None,
                'maximum': None,
                'minimum': None,
                
                # Features chacges
                'mean_abs_change': None,
                'mean_change': None,
                'abs_energy': None,
                'absolute_sum_of_changes': None,
                
                # Trend 
                'linear_trend': [{'attr': 'slope'}, {'attr': 'rvalue'}, {'attr': 'intercept'}],
                'augmented_dickey_fuller': [{'attr': 'teststat'}, {'attr': 'pvalue'}],
                
                # Entropy 
                'binned_entropy': [{'max_bins': 10}],
                'approximate_entropy': [{'m': 2, 'r': 0.1}],
                
                # Autocorrelation
                'autocorrelation': [{'lag': 1}, {'lag': 2}, {'lag': 5}, {'lag': 10}],
                'partial_autocorrelation': [{'lag': 1}, {'lag': 2}],
                
                # Fourier features
                'fft_coefficient': [{'coeff': 0, 'attr': 'abs'}, 
                                {'coeff': 1, 'attr': 'abs'}],
                
                # Non-linear features
                'cid_ce': [{'normalize': True}],
                'number_peaks': [{'n': 1}, {'n': 5}],  
            }
        else:
            self.fc_parameters = default_fc_parameters
        
        self.data = data.copy()
        self.features_df = None
        self.selected_features = None
        self.id_column = 'id'
        
        # 1. Convert all numeric columns to float
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # 2. Check for NaN values
        nan_count = self.data.isna().sum().sum()
        if nan_count > 0:
            self.data = self.data.ffill().bfill()  # Forward then backward fill
        
        # 3. Remove non-numeric columns that might cause issues
        non_numeric_cols = self.data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            self.data = self.data.select_dtypes(include=[np.number])
        
    def prepare_tsfresh_data(self, id_column='id'):
        """
        Prepare data for TSfresh feature extraction
        
        Parameters:
        id_column (str): Name of the ID column
        
        """
        
        print("="*50)
        print("Preparing data for TSfresh...")
        print("="*50)
        
        df = self.data.copy()
        
        # Create ID column if not exist
        if id_column not in df.columns:
            df[id_column] = 1 
        
        # Create time column for sorting
        if 'time_ts' not in df.columns:
            df['time_ts'] = range(len(df))
        
        self.id_column = id_column
        return df
        
    def extract_features(self, column_names=None, id_column='id'):
        
        """
        Extract features using TSfresh
        
        Parameters:
        column_names (list): List of column names to extract features from
        id_column (str): Name of the ID column
        
        """
        
        print("="*50)
        print("Extracting features with TSfresh...")
        print("="*50)
        
        # Prepare data
        prepared_df = self.prepare_tsfresh_data(id_column)
        
        # Extract features
        features = extract_features(
            prepared_df, 
            column_id = id_column, 
            column_sort = 'time_ts',
            default_fc_parameters = self.fc_parameters,
            impute_function = impute,
            n_jobs = 1  # Use all available cores
        )
        
        self.features_df = features
        print(f"Extracted {features.shape[1]} features from {len(column_names)} columns")
        print("="*50)
        
        return features
    
    def select_relevant_features(self, target, fdr_level=0.05):
        
        """
        Select relevant features using TSfresh
        
        Parameters:
        target: Target variable series
        fdr_level (float): FDR level for feature selection
        
        """
        
        if self.features_df is None:
            raise ValueError("Must extract features first using extract_features()")
        
        print("="*50)
        print("Selecting relevant features...")
        print("="*50)
        
        # Select relevant features
        features_filtered = select_features(
            self.features_df, 
            target,
            fdr_level=fdr_level
        )
        
        self.selected_features = features_filtered.columns.tolist()
        
        print(f"Selected {len(self.selected_features)} relevant features (FDR level: {fdr_level})")
        print("="*50)
        
        return features_filtered
    
    def fast_extract_features(self, 
                              target_column=None,
                              id_column='id'):
        
        """
        Quick feature extraction pipeline
        
        Parameters:
        target_column (str): Name of target column for feature selection
        id_column (str): Name of the ID column
        
        """
        
        print("="*50)
        print("Running fast extraction pipeline...")
        print("="*50)
        
        print("Preparing data for TSfresh...")
        prepared_data = self.prepare_tsfresh_data(id_column = id_column)
        
        # Extract features
        print("Extracting features...")
        features = self.extract_features(id_column = id_column)
        
        # Select relevant features if target is provided
        if target_column is not None and target_column in self.data.columns:
            print("Selecting relevant features...")
            target = self.data[target_column]
            features = self.select_relevant_features(target)
        
        return features
    
    # def save_features(self, filepath: str = '../models/features_ts.joblib'):
        
    #     """
    #     Save features to a file
        
    #     Parameters:
    #     filepath (str): Path to save the features
        
    #     """
        
    #     if self.features_df is not None:
    #         # Create directory if it doesn't exist
    #         import os
    #         os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
    #         joblib.dump({
    #             'features_df': self.features_df,
    #             'selected_features': self.selected_features,
    #             'fc_parameters': self.fc_parameters,
    #             'data_columns': self.data.columns.tolist()
    #         }, filepath)
    #         print(f"Features saved to {filepath}")
    #     else:
    #         print("No features to save. Run extract_features() first.")
    
    # def load_features(self, filepath: str):
        
    #     """
    #     Load features from a file
        
    #     Parameters:
    #     filepath (str): Path to load features from
        
    #     """
        
    #     try:
    #         data = joblib.load(filepath)
    #         self.features_df = data['features_df']
    #         self.selected_features = data['selected_features']
    #         self.fc_parameters = data['fc_parameters']
    #         print(f"Features loaded from {filepath}")
    #         return self.features_df
    #     except Exception as e:
    #         print(f"Error loading features: {e}")
    #         return None

    def get_feature_stats(self):
        
        """
        Get statistics about the extracted features
        
        """
        
        if self.features_df is None:
            print("No features extracted yet")
            return None
        
        stats = {
            'total_features': self.features_df.shape[1],
            'selected_features': len(self.selected_features) if self.selected_features else 0,
            'nan_percentage': (self.features_df.isna().sum().sum() / 
                             (self.features_df.shape[0] * self.features_df.shape[1])) * 100
        }
        
        print("="*50)
        print("FEATURE STATISTICS")
        print("="*50)
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        print("="*50)
        
        return stats