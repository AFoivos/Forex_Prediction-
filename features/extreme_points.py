import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
from scipy.signal import argrelextrema

import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from plots import ForexPlotter

import warnings
warnings.filterwarnings('ignore')

class ForexExtremePoints:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
        prints: bool = True,
    ):

        """
        Class for detecting extreme points (swing highs and lows)

        Parameters:
        data (pd.DataFrame): DataFrame containing the data
        close_col (str): Column name for close price
        prints (bool): Whether to print information
        last_period (List[int]): List of periods to consider for plotting
        order_nums (List[int]): List of order numbers for swing point detection
        
        """
        
        self.prints = prints
        
        if self.prints:
            print("="*50)
            print("EXTREME POINT DETECTION")
            print("="*50)
            print("Available function: \n1 detect_extreme_points")
            print("="*50)
        
        self.data = data.copy()
        self.close_col = close_col
        
        self.parameters = {}
        
        self.extreme_data = pd.DataFrame(
            {self.close_col: self.data[self.close_col]},
            index=self.data.index
        )
        
        self._validate_columns()
        
    def _validate_columns(
            self, 
            columns: list[str] = None,
        ):
            if columns is None:
                columns = [self.close_col]
            
            missing_cols = [col for col in columns if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in data: {missing_cols}")
        
    def detect_extreme_points(
        self, 
        periods: List[int] = [5, 10, 20, 50, 100, 200, 300, 400, 500, 600],
        orders: List[int] = [1, 2, 3, 4, 5, 10, 20, 25, 30, 35, 40, 45, 50]
    ):
        
        self._validate_columns()
        
        self.parameters['periods'] = periods
        self.parameters['orders'] = orders
        
        prices = self.data[self.close_col].values

        for period in periods:
            for order in orders:
                local_max = argrelextrema(
                    prices, 
                    np.greater,
                    order=order
                )[0]
                local_min = argrelextrema(
                    prices,
                    np.less,
                    order=order
                )[0]

                labels = np.zeros(len(prices))
                labels[local_min] = 1  
                labels[local_max] = 3   

                labeled_points = [(i, labels[i]) for i in range(len(labels)) if labels[i] in [1, 3]]

                for idx in range(len(labeled_points) - 1):
                    i1, l1 = labeled_points[idx]
                    i2, l2 = labeled_points[idx + 1]

                    if l1 == 1 and l2 in [1, 3]:  
                        labels[i1 + 1:i2] = 2     
                    elif l1 == 3 and l2 in [3, 1]: 
                        labels[i1 + 1:i2] = 4    

                self.extreme_data[f'Label_p{period}_o{order}'] = labels.astype(int)

        count_removed_rows = self.data.shape[0] - self.extreme_data.shape[0]

        if self.prints:
            print('='*50)
            print('Data Info')
            print(self.extreme_data.info())
            print('='*50)
            print(f'Shape of data {self.extreme_data.shape}')
            print('='*50)
            print(f'{count_removed_rows} rows removed')
            print('='*50)
            
        return self.extreme_data, self.parameters