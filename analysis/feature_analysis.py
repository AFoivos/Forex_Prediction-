import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor

from xgboost import XGBRegressor, XGBClassifier

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

class ForexFeatureAnalysis:
    def __init__(self, 
                 data: pd.DataFrame,
                 target_col: str = 'close',
                 use_xgboost: bool = False,
                 scale: bool = True):
        
        """
        Feature Analysis Class
        !!!! DONT FORGET TO HANDLE MISSING VALUES BEFORE YOU USE THIS CLASS !!!!
        
        Parameters:
        data (pd.DataFrame): DataFrame with features
        target_col (str): Target column for predictive analysis
        
        """
        
        self.data = data.copy()
        self.target_col = target_col
        self.results = pd.DataFrame()
        self.use_xgboost = use_xgboost
        self.scale = scale
        if self.scale:
            self.scaler = StandardScaler()

        
        # Numeric cols
        self.numeric_cols = [
                col for col in self.data.columns 
                if pd.api.types.is_numeric_dtype(self.data[col])
            ]        
        
        # Remove target col f
        if target_col in self.numeric_cols:
            self.numeric_cols.remove(target_col)
        
        print("="*50)
        print("FEATURE ANALYSIS")
        print("="*50)
        print(f"Target column: {self.target_col}")
        print(f"Feature numeric columns: {self.numeric_cols}")
        print("="*50)

    def calculate_vif(self, threshold: float = 10.0):
        
        """
        Calculate Variance Inflation Factor for multicollinearity
        
        Parameters:
        threshold (float): VIF threshold for high multicollinearity
        
        Returns:
        DataFrame with VIF scores
        
        """
        
        print("="*50)
        print(" Calculating VIF for Multicollinearity Check...")
        print("="*50)
        
        # Select only numeric columns without missing values
        vif_data = self.data[self.numeric_cols]
        
        # Calculate VIF for each feature
        vif_scores = []
        for i, col in enumerate(self.numeric_cols):
            try:
                vif = variance_inflation_factor(vif_data.values, i)
                vif_scores.append({'Feature': col, 'VIF': vif})
            except:
                vif_scores.append({'Feature': col, 'VIF': np.nan})
        
        vif_df = pd.DataFrame(vif_scores)
        vif_df['High_Multicollinearity'] = vif_df['VIF'] > threshold
        
        # Display results
        print(f"VIF Analysis (Threshold: {threshold}):")
        print("=" * 50)
        for index, row in vif_df.iterrows():
            
            if row["High_Multicollinearity"]:
                status = "HIGH"
            else:
                status = "OK"
            print(f"{row['Feature']:30} | VIF: {row['VIF']:6.2f} | {status}")
        
        high_vif_count = vif_df['High_Multicollinearity'].sum()
        print(f"Found {high_vif_count} features with high multicollinearity")
        
        return vif_df
        
        # Correlation with target
        correlations = []
        
        for col in self.numeric_cols:
            corr = analysis_data[col].corr(analysis_data[self.target_col])
            correlations.append({'Feature': col, 'Correlation': corr})
        
        corr_importance = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)
        results['correlation'] = corr_importance
        
        # Store results
        if self.use_xgboost:
            model_name = "XGBoost"
        else:
            model_name = "RandomForest"
            
        self.results[f'{model_name}_Importance'] = model_importance.set_index('Feature')['Model_Importance']
        self.results['MI_Score'] = mi_importance.set_index('Feature')['MI_Score']
        self.results['Correlation'] = corr_importance.set_index('Feature')['Correlation']
        
        
        # Display results
        print("="*50)
        print("Feature Importance Analysis Results:")
        print("="*50)
        print("Top 10 Features by Random Forest Importance:")
        print(model_importance.head(10).to_string(index=False))
        print("="*50)
        print("Top 10 Features by Mutual Information:")
        print(mi_importance.head(10).to_string(index=False))
        print("="*50)
        print("Top 10 Features by Absolute Correlation:")
        print(corr_importance.head(10).to_string(index=False))
        print("="*50)
        
        return results

    def feature_relationships(self, top_n: int = 15):
        
        """
        Analyze relationships between features
        
        Parameters:
        top_n (int): Number of top features to analyze
        
        """
        
        print("="*50)
        print(f"Analyzing Feature Relationships (Top {top_n} features)...")
        print("="*50)
        
        numeric_data = self.data[self.numeric_cols]
        
        # Calculate variance and select top features
        variances = numeric_data.var().sort_values(ascending=False)
        top_features = variances.head(top_n).index.tolist()
        
        # Correlation matrix
        corr_matrix = numeric_data[top_features].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title(f'Correlation Matrix of Top {top_n} Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
        
        print(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.8):")
        for pair in high_corr_pairs:
            print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
        
        return {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_corr_pairs
        }
    
    def lag_analysis(self, max_lags: int = 24): # Full Day Lags  
        
        """
        Calculate lags for features
        
        Parameters:
        max_lags (int): Maximum number of lags to test
        
        """
        
        print("="*50)
        print("Analyzing Lags...")
        print("="*50)
        
        print(f"Optimal Lag Analysis (Max Lags: {max_lags})...")
        
        results = {}
        
        for feature in self.numeric_cols:
            
            # Calculate correlations at different lags
            lag_correlations = []
            for lag in range(0, max_lags + 1):
                
                if lag == 0:
                    corr = self.data[feature].corr(self.data[self.target_col])
                else:
                    corr = self.data[feature].shift(lag).corr(self.data[self.target_col])
                lag_correlations.append({'Lag': lag, 'Correlation': corr})
            
            lag_df = pd.DataFrame(lag_correlations)
            optimal_lag = lag_df.loc[lag_df['Correlation'].abs().idxmax()]
            
            results[feature] = {
                'optimal_lag': optimal_lag['Lag'],
                'max_correlation': optimal_lag['Correlation'],
                'all_lags': lag_df
            }
        
        # Display top features by absolute correlation
        feature_performance = []
        for feature, result in results.items():
            feature_performance.append({
                'Feature': feature,
                'Optimal_Lag': result['optimal_lag'],
                'Correlation': result['max_correlation']
            })
        
        performance_df = pd.DataFrame(feature_performance)
        performance_df['Abs_Correlation'] = performance_df['Correlation'].abs()
        performance_df = performance_df.sort_values('Abs_Correlation', ascending=False)
        
        print("Top Features by Predictive Power:")
        print(performance_df.head(15).to_string(index=False))
        
        return results
    
    def comprehensive_analysis(self, 
                               target_col: str = "close", 
                               problem_type: str = 'regression'):
        
        """
        Run comprehensive feature analysis
        
        Parameters:
        target_col (str): Target column name
        problem_type (str): Type of problem ('regression' or 'classification')
        
        """
        
        print("="*50)
        print("Starting Full Feature Analysis...")
        print("="*50)
        
        results = {}
        
        results['vif'] = self.calculate_vif()
        results['importance'] = self.feature_importance_analysis(problem_type)
        results['relationships'] = self.feature_relationships()
        results['lags'] = self.optimal_lag_analysis()
        
        print("="*50)
        print("Feature Analysis Complete!")
        print("="*50)
        
        return results