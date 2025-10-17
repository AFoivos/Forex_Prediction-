import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

class TimeSeriesAutoencoder:
    def __init__(
        self,
        input_dim,
        encoding_dim: int = 2,
        learning_rate: float = 0.001,
        backend_clear: bool = True,
    ):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.autoencoder = None
        self.encoder = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.backend_clear = backend_clear
        
    def build_model(self):
        
        """
        Build the autoencoder architecture
        
        """
        
        if self.backend_clear:
            tf.keras.backend.clear_session()
        
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(128,activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        encoded = Dense(8, activation='relu')(encoded)
        bottleneck = Dense(self.encoding_dim, activation='linear')(encoded)
        
        decoded = Dense(8, activation='relu')(bottleneck)
        decoded = Dense(16, activation='relu')(decoded)
        decoded = Dense(32, activation='relu')(decoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dropout(0.2)(decoded)
        output_layer = Dense(self.input_dim, activation='linear')(decoded)
        
        self.autoencoder = Model(input_layer, output_layer)
        self.encoder = Model(input_layer, bottleneck)
        
        self.autoencoder.compile(
            optimizer = Adam(learning_rate=self.learning_rate), 
            loss = 'mse', 
            metrics = ['mae']
        )
        
    def prepare_data(
        self, 
        features_df: pd.DataFrame,
        target_column: str = 'close',
        test_size: float = 0.2
    ):
        """
        Prepare and split data maintaining temporal order
        
        """
        
        self.features = features_df.drop(target_column, axis=1).values
        self.target = features_df[target_column].values
        self.feature_names = features_df.drop(target_column, axis=1).columns.tolist()
        
        self.scaled_features = self.scaler.fit_transform(self.features)
        
        self.train_size = int((1 - test_size) * len(self.scaled_features))
        self.train_data = self.scaled_features[:self.train_size]
        self.test_data = self.scaled_features[self.train_size:]
        
        print(f"Training samples: {len(self.train_data)}")
        print(f"Test samples: {len(self.test_data)}")
        
        return self.train_data, self.test_data
    
    def train(
        self, 
        epochs: int = 100,
        batch_size: int = 32, 
        verbose: int = 1
    ):
        """
        Train the autoencoder
        
        """
        
        if self.autoencoder is None:
            self.build_model()
            
        self.history = self.autoencoder.fit(
            self.train_data,
            self.train_data,
            epochs = epochs,
            batch_size = batch_size,
            validation_data = (self.test_data, self.test_data),
            verbose = verbose,
            shuffle = False  
        )
        
        self.is_trained = True
        return self.history
    
    def encode(
        self,
        data = None 
    ):
        
        """
        Encode data to latent space
        
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before encoding")
            
        if data is None:
            data = self.scaled_features
            
        return self.encoder.predict(data, verbose=0)
    
    def add_latent_features(
        self,
        original_df
    ):
    
        """
        Add latent dimensions to original dataframe
        
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before adding latent features")
            
        encoded_features = self.encode()
        
        df_with_latent = original_df.copy()
        df_with_latent['latent_dim_1'] = encoded_features[:, 0]
        df_with_latent['latent_dim_2'] = encoded_features[:, 1]
        
        return df_with_latent
    
    def evaluate(self):
        
        """
        Evaluate model performance
        
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        train_reconstruction = self.autoencoder.predict(self.train_data, verbose=0)
        test_reconstruction = self.autoencoder.predict(self.test_data, verbose=0)
        
        train_error = np.mean(np.square(self.train_data - train_reconstruction))
        test_error = np.mean(np.square(self.test_data - test_reconstruction))
        
        metrics = {
            'final_training_loss': self.history.history['loss'][-1],
            'final_validation_loss': self.history.history['val_loss'][-1],
            'training_reconstruction_error': train_error,
            'test_reconstruction_error': test_error,
            'generalization_gap': test_error - train_error
        }
        
        return metrics
    
    def plot_training_history(self):
        
        """
        Plot training history
        
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting")
            
        plt.figure(figsize=(10, 4))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def plot_latent_space(self, df_with_latent, color_by='close'):
        
        """
        Plot 2D latent space representation
        
        """
        
        plt.figure(figsize=(12, 8))
        
        latent_1 = df_with_latent['latent_dim_1'].values
        latent_2 = df_with_latent['latent_dim_2'].values
        color_values = df_with_latent[color_by].values
        
        scatter = plt.scatter(
            latent_1[:self.train_size], 
            latent_2[:self.train_size], 
            c = color_values[:self.train_size], 
            cmap = 'viridis', 
            alpha = 0.7, 
            label = 'Train', 
            s = 20
            )
        scatter = plt.scatter(
            latent_1[self.train_size:],
            latent_2[self.train_size:], 
            c = color_values[self.train_size:], 
            cmap = 'plasma', 
            alpha = 0.7, 
            marker = 'x',
            label = 'Test', 
            s = 20
        )
        
        plt.colorbar(scatter, label=color_by)
        plt.title('Autoencoder (Train/Test Split)')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.legend()
        plt.show()
    
    def analyze_clusters(
        self, 
        df_with_latent: pd.DataFrame, 
        n_clusters: int = 3 
    ):
        
        """
        Perform clustering analysis in latent space
        
        """
        
        latent_features = df_with_latent[['latent_dim_1', 'latent_dim_2']].values
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(latent_features)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            latent_features[:, 0],
            latent_features[:, 1], 
            c = clusters, 
            cmap = 'Set1', 
            alpha = 0.7
        )
        plt.colorbar(scatter, label='Cluster')
        plt.title('K-means Clustering in Latent Space')
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.show()
        
        # Cluster analysis
        df_with_clusters = df_with_latent.copy()
        df_with_clusters['cluster'] = clusters
        
        cluster_stats = df_with_clusters.groupby('cluster').agg({
            'close': ['mean', 'std', 'count'],
            'latent_dim_1': ['mean', 'std'],
            'latent_dim_2': ['mean', 'std']
        }).round(4)
        
        return df_with_clusters, cluster_stats

    def compresive_run(
        self,
    ):
        pass