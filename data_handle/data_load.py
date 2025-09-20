import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import date2num
import mplfinance as mpf

import warnings
warnings.filterwarnings('ignore')

class ForexDataLoad:
    def __init__(self, 
                 file_path: str = None, 
                 data: pd.DataFrame = None, 
                 connection_string: str = None, 
                 query: str = None ,
                 prints: bool = True
                 ):
        
        """
        Initialize the Forex Data Loader
        Parameters:
        prints (bool): Whether to print loading information

        """
        
        print("="*50)
        print("FOREX DATA LOADER")
        print("="*50)
        print(" Available Fuctions \n1 load_csv \n2 load_dataframe \n3 load_from_database")
        print("="*50)
        
        self.file_path = file_path
        self.data = data
        self.connection_string = connection_string
        self.query = query
        self.data = None
        self.prints = prints
        
        try:
            if self.file_path is not None:
                self.load_csv()
            elif self.data is not None:
                self.load_dataframe()
            elif self.connection_string is not None and self.query is not None:
                self.load_from_database()
            else:
                print("No data loaded yet. Please load data using one of the methods.")
        except Exception as e:
            print(f"Error during initialization: {e}")

        
    def load_csv(self):
        
        """
        Load data from a CSV file
        !!! MUST BE FROM METATRADER 5 !!!
        Parameters:
        file_path (str): Path to the CSV file
        
        """ 
        
        try:
            # Load the CSV file
            new_column_names = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'volume', 'spread']
            self.data = pd.read_csv(self.file_path,
                                        header=None,   
                                        sep='\t',
                                        names=new_column_names, 
                                        skiprows=1  
                                        )
            
            self.data['datetime'] = pd.to_datetime(self.data['date'] + ' ' + self.data['time'])
            
            # Remove unnecessary columns
            self.data.drop(columns=['tickvol',  
                            'spread',
                            'date',
                            'time'],
                            inplace=True
                            )
    
            # self.data = self.data[self.data['datetime'].dt.year > 2020]
    
            self.data.set_index('datetime', inplace=True)
            
            if self.prints:
                # Display initial information
                print('Data loaded successfully!')
                print(f'Shape: {self.data.shape}')
                print("\n" + "="*50)
        except Exception as e:
            raise ValueError(f"Error loading file: {e}")
    
    def load_dataframe(self):
        
        """
        Load data from a DataFrame
        Parameters:
        data (pd.DataFrame): DataFrame containing the data
        
        """ 
        
        try:
            if isinstance(self.data, pd.DataFrame):
                self.data = self.data.copy()
                if self.prints:
                    print('Data loaded successfully from DataFrame!')
                    print(f'Shape: {self.data.shape}')
                    print("\n" + "="*50)
            else:
                raise ValueError("Input is not a valid DataFrame.")
        except Exception as e:
            raise ValueError(f"Error loading DataFrame: {e}")
            
    def load_from_database(self, 
                           connection_string: str , 
                           query: str
                           ):
        
        # """
        # Load data from a database
        # Parameters:
        # connection_string (str): Database connection string
        # query (str): SQL query to fetch the data
        #
        # """
        # 
        # try:
        #     import sqlalchemy
        #     engine = sqlalchemy.create_engine(connection_string)
        #     self.data = pd.read_sql(query, engine)
        #     if self.prints == True:
        #         print('Data loaded successfully from database!')
        #         print(f'Shape: {self.data.shape}')
        #         print("\n" + "="*50)
        # except Exception as e:
        #     print(f"Error loading data from database: {e}")
        
        pass