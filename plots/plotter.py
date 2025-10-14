import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexPlotter:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
    ):
        self.data = data.copy()
        self.close_col = close_col
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
    
    def plot_signals(
        self,
        periods: int = 2400,
        cols: List[str] = None 
    ):
        
        if cols is None:
            cols = self.data.drop(columns = self.close_col).columns
        else:
            if self.close_col in cols:
                cols = cols.tolist()
                cols.remove(self.close_col)
            
        self._validate_columns(columns = cols)
            
        plot_data = self.data.copy().tail(periods)
            
        for col in cols:    
            
            plt.figure(figsize=(15, 8))

            colors = ['blue' if x == 0 else 'red' if x == 1 else 'green' for x in plot_data[col]]

            for i in range(len(plot_data) - 1):
                plt.plot(
                    plot_data.index[i:i+2],
                    plot_data[self.close_col].iloc[i:i+2], 
                    color=colors[i], 
                    linewidth=2
                )

            for color, value in [ ('red', 1), ('green', 2)]:
                mask = plot_data[col] == value
                if mask.any():
                    plt.scatter(
                        plot_data.index[mask],
                        plot_data[self.close_col][mask], 
                        color = color, 
                        s = 50, 
                        label = f'Cross = {value}'
                    )
                    
            plt.title(col, fontsize=20)
            plt.xlabel('DateTime', fontsize=12)
            plt.ylabel(self.close_col, fontsize=12)
            plt.legend()
            plt.grid(True, alpha = 0.3)
            plt.tight_layout()
            plt.show()
    
    def visualize_swing_points(
        self, 
        cols: str = None,
        figsize: tuple = (16, 8),
        periods: int = 2400,
    ):
        
        """
        Visualize swing points for a specific label column
        
        Parameters:
        label_column (str): The exact column name (e.g., 'Label_p50_o5')
        last_bars (int): Number of last bars to visualize
        figsize (tuple): Figure size
        
        """
        if cols is None:
            cols = self.data.drop(columns = self.close_col).columns
        else:
            if self.close_col in cols:
                cols = cols.tolist()
                cols.remove(self.close_col)
    
        self._validate_columns(columns = cols)
        
        plot_data = self.data.copy().tail(periods)
        
        for col in cols:
            plt.figure(figsize=figsize)
            plt.plot(plot_data[self.close_col], label=self.close_col, color='blue', linewidth=1)

            buy_signals = plot_data[plot_data[col] == 1]  
            sell_signals = plot_data[plot_data[col] == 3]  

            plt.scatter(buy_signals.index, buy_signals[self.close_col],
                        marker='^', color='green', s=200, label='Lower Extreme', zorder=5)

            plt.scatter(sell_signals.index, sell_signals[self.close_col],
                        marker='v', color='red', s=200, label='Upper Extreme', zorder=5)

            plt.title(f"Swing Points - {col} (Last {periods} Bars)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
