import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexADXSignals:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
    ):
        """
        Class for ADX signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
        print("="*50)
        print("ADX SIGNAL GENERATION")
        print("="*50)
        
        self.close_col = close_col
        self.data = data.copy()
        
        self.signals = pd.DataFrame(
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
            
        """
        
        required_cols = [
            self.close_col,
        ]
        
        if columns is not None:
            required_cols.extend(columns)
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")