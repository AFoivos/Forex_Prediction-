import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexPlotter:
    def __init__(self, 
                 style: str = 'seaborn-v0_8', 
                 figsize: tuple = (15, 8),
                 colors: List[str] = None
                 ):
        
        """
        Initialize the ForexPlotter with default style, figure size, and color palette.
        
        Parameters:
        style (str): Matplotlib style to use for plots.
        figsize (tuple): Default figure size for plots.
        colors (List[str]): List of colors to use for plotting multiple series.
        
        """
        
        plt.style.use(style)
        self.figsize = figsize
        self.colors = colors or ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
                               ]
        
    def line_plot(self, 
                data: Union[pd.DataFrame, pd.Series],
                title: str = 'Time Series Plot',
                ylabel: str = 'Value',
                xlabel: str = 'Date',
                grid: bool = True,
                alpha: float = 0.8,
                linewidth: float = 1.5,
                save_path: Optional[str] = None
                ) -> plt.Figure:
        
        """
        Create a line plot for a given DataFrame or Series.
        
        Parameters:
        data (pd.DataFrame or pd.Series): Data to plot.
        title (str): Title of the plot.
        ylabel (str): Label for the y-axis.
        xlabel (str): Label for the x-axis.
        grid (bool): Whether to display grid lines on the plot.
        alpha (float): Transparency level for plot lines.
        linewidth (float): Width of the plot lines.
        save_path (str): Path to save the plot image.
        
        """
        
        fig, ax = plt.subplots(figsize=self.figsize)
    
        if isinstance(data, pd.Series):
            ax.plot(data.index, 
                    data.values, 
                    color=self.colors[0], 
                    linewidth=linewidth, 
                    alpha=alpha,
                    label=data.name if data.name else 'Value'
                    )
            
        elif isinstance(data, pd.DataFrame):
            for i, column in enumerate(data.columns):
                
                ax.plot(data.index, 
                        data[column].values,
                        color=self.colors[i % len(self.colors)],
                        linewidth=linewidth,
                        alpha=alpha,
                        label=column
                        )
            
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=12)
        
        if grid:
            ax.grid(True, alpha=0.3)
        
        if isinstance(data, pd.DataFrame) or (isinstance(data, pd.Series) and data.name):
            ax.legend(loc='best')
        
        plt.tight_layout()
        plt.show()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
        
    def multiple_series_plot(self,
                            series_dict: Dict[str, pd.Series],
                            title: str = 'Multiple Series Comparison',
                            ylabel: str = 'Value',
                            xlabel: str = 'Date',
                            subplots: bool = False,
                            normalize: bool = False,
                            save_path: Optional[str] = None,
                            ylabel_str = None
                            ) -> plt.Figure:
        
        """
        Create line plots for multiple time series, either in a single plot or as subplots.
        
        Parameters:
        series_dict (Dict[str, pd.Series]): Dictionary of series to plot with keys as labels.
        title (str): Title of the plot. If subplots is True, this will be the suptitle.
        ylabel (str): Label for the y-axis.
        xlabel (str): Label for the x-axis.
        subplots (bool): Whether to plot each series in its own subplot.
        normalize (bool): Whether to normalize series to start at 100.
        save_path (str): Path to save the plot image.
        
        """
        
        if normalize:
            series_dict = {name: (series / series.iloc[0] * 100) 
                          for name, series in series_dict.items()}
            ylabel = 'Normalized Value (Base=100)'
        
        if subplots:
            n_series = len(series_dict)
            fig, axes = plt.subplots(n_series, 1,
                                     figsize=(self.figsize[0], 
                                     self.figsize[1] * n_series/2)
                                     )
            
            if n_series == 1:
                axes = [axes]
                
                for ax, ((name, series), color) in zip(axes, zip(series_dict.items(), self.colors)):
                    ax.plot(series.index,
                            series.values, 
                            color=color, 
                            linewidth=2, 
                            label=name
                            )
                    
                    ax.set_title(name, 
                                 fontsize=14
                                 )
                    
                    ax.set_ylabel(ylabel)
                    ax.grid(True, 
                            alpha=0.3
                            )
                    
                    ax.legend()
            
            plt.suptitle(title, 
                         fontsize=16, 
                         fontweight='bold'
                         )
        else:
            
            fig, ax = plt.subplots(figsize=self.figsize)
            
            for (name, series), color in zip(series_dict.items(), self.colors):
                
                ax.plot(series.index, 
                        series.values, 
                        color=color, 
                        linewidth=2, 
                        label=name
                        )
            
            ax.set_title(title, 
                        fontsize=16, 
                        fontweight='bold'
                        )
            
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
        
        plt.tight_layout()
        plt.show()
        
        if save_path:
            plt.savefig(save_path,
                        dpi=300, 
                        bbox_inches='tight'
                        )
        
        return fig
                
    def rolling_statistics_plot(self,
                               series: pd.Series,
                               windows: List[int] = [20, 50, 100],
                               statistics: List[str] = ['mean', 'std'],
                               title: str = 'Rolling Statistics',
                               save_path: Optional[str] = None,
                               xlabel: str = None
                               ) -> plt.Figure:    
        
        """
        Plot rolling statistics (mean, std, min, max) for a given time series.  
        
        Parameters:
        series (pd.Series): Time series data to plot.
        windows (List[int]): List of window sizes for rolling calculations.
        statistics (List[str]): List of statistics to plot. Options: 'mean', 'std', 'min', 'max'.
        title (str): Title of the plot.
        save_path (str): Path to save the plot image.
        xlabel (str): Label for the x-axis.
        
        """
        
        fig, axes = plt.subplots(len(statistics), 1, 
                               figsize=(self.figsize[0], 
                               self.figsize[1] * len(statistics)/2)
                               )
        
        if len(statistics) == 1:
            axes = [axes]
        
        for ax, stat in zip(axes, statistics):
            for i, window in enumerate(windows):
                if stat == 'mean':
                    rolling_data = series.rolling(window=window).mean()
                elif stat == 'std':
                    rolling_data = series.rolling(window=window).std()
                elif stat == 'min':
                    rolling_data = series.rolling(window=window).min()
                elif stat == 'max':
                    rolling_data = series.rolling(window=window).max()
                else:
                    continue
                
                ax.plot(rolling_data.index, 
                        rolling_data.values,
                        color=self.colors[i % len(self.colors)],
                        linewidth=2,
                        label=f'Window {window}'
                        )
            
            ax.set_title(f'Rolling {stat.upper()}', fontsize=14)
            ax.set_ylabel(stat.upper())
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle(f'{title} - {series.name}', 
                     fontsize=16, 
                     fontweight='bold'
                     )
        
        plt.tight_layout()
        plt.show()
        
        if save_path:
            plt.savefig(save_path, 
                        dpi=300, 
                        bbox_inches='tight'
                        )
        
        return fig
    
    def correlation_heatmap(self,
                           data: pd.DataFrame,
                           title: str = 'Correlation Heatmap',
                           annot: bool = True,
                           cmap: str = 'coolwarm',
                           save_path: Optional[str] = None
                           ) -> plt.Figure:
        
        """
        Plot a correlation heatmap for the given DataFrame.
        
        Parameters:
        data (pd.DataFrame): Data to plot.
        title (str): Title of the plot.
        annot (bool): Whether to display values in the heatmap.
        cmap (str): Colormap for the heatmap.
        save_path (str): Path to save the plot image.
        
        """
        
        corr_matrix = data.corr()
            
        fig, ax = plt.subplots(figsize=(max(10, len(data.columns)*0.8), 
                                        max(8, len(data.columns)*0.6)))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    annot=annot,
                    cmap=cmap,
                    center=0,
                    square=True,
                    fmt='.3f',
                    ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        if save_path:
            plt.savefig(save_path, 
                        dpi=300,
                        bbox_inches='tight'
                        )
        
        
        return fig    
    
    def plot_technical_indicators(self,
                                 data: pd.DataFrame,
                                 price_column: str = 'close',
                                 indicator_columns: List[str] = None,
                                 title: str = 'Technical Indicators',
                                 save_path: Optional[str] = None
                                 ) -> plt.Figure:    
        
        """
        Plot price along with multiple technical indicators.
        
        Parameters:
        data (pd.DataFrame): Data containing price and technical indicators.
        price_column (str): Column name for the price data. Default is 'close'.
        indicator_columns (List[str]): List of column names for technical indicators. If None, all columns except price_column are used.
        title (str): Title of the plot.
        save_path (str): Path to save the plot image.
        
        """
        
        if indicator_columns is None:
            indicator_columns = [col for col in data.columns if col != price_column and col not in ['open', 'high', 'low']]
        
        n_plots = 1 + len(indicator_columns)
        fig, axes = plt.subplots(n_plots, 1, 
                                 figsize=(self.figsize[0],
                                 self.figsize[1] * n_plots/2)
                                 )
        
        if n_plots == 1:
            axes = [axes]
        
        # Plot price
        axes[0].plot(data.index, 
                     data[price_column], 
                     color=self.colors[0], 
                     linewidth=2, 
                     label=price_column
                     )
        
        axes[0].set_title(f'{price_column.upper()} Price', fontsize=14)
        axes[0].set_ylabel('Price')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot indicators
        for i, indicator in enumerate(indicator_columns, 1):
            
            axes[i].plot(data.index, 
                         data[indicator], 
                         color=self.colors[i % len(self.colors)], 
                         linewidth=2, 
                         label=indicator
                         )
            
            axes[i].set_title(indicator.upper(), fontsize=14)
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        if save_path:
            plt.savefig(save_path, 
                        dpi=300, 
                        bbox_inches='tight'
                        )
        
        return fig