import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple, Optional, Dict, Any

class ForexAutoencoder:
    def __init__(
        self,
        data: pd.DataFrame,
        input_dim: int,
        target_col: str = 'close',
        encoding_dim: int = 2,
        prefered_name: str = 'dim',
        epochs: int = 100,
        batch_size: int = 32,
        test_size: float = 0.2, 
        verbose: int = 1,   
        learning_rate: float = 0.001,
        backend_clear: bool = True,
        prints: bool = True,
        hidden_dims: list = [128, 64, 32, 16, 8],
        random_state: int = 42,
        use_callbacks: bool = True
    ):
        
        self.data = data.copy()
        self.target_col = target_col
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.prefered_name = prefered_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_size = test_size
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.backend_clear = backend_clear
        self.prints = prints
        self.hidden_dims = hidden_dims
        self.random_state = random_state
        self.use_callbacks = use_callbacks
        
        self._prepare_data_structures()
        
        self.autoencoder = None
        self.encoder = None
        self.scaler = StandardScaler()
        
    def _prepare_data_structures(
        self
    ):
        
        self.new_data = pd.DataFrame(
            {self.target_col: self.data[self.target_col]},
            index=self.data.index
        )
        
        if self.target_col in self.data.columns:
            self.features = self.data.drop(self.target_col, axis=1).values
            self.feature_names = self.data.drop(self.target_col, axis=1).columns.tolist()
        else:
            self.features = self.data.values
            self.feature_names = self.data.columns.tolist()
            
        self.target = self.data[self.target_col].values if self.target_col in self.data.columns else None
        
    def model_build(self, hidden_dims: list = [128, 64, 32, 16, 8]):
        if self.backend_clear:
            tf.keras.backend.clear_session()
            
        # Input layer
        input_layer = Input(shape=(self.input_dim,))
        x = input_layer
        
        # Encoder
        for dim in hidden_dims:
            x = Dense(dim, activation = 'relu')(x)
            x = BatchNormalization()(x) 
            x = Dropout(0.2)(x)
        
        # Bottleneck
        bottleneck = Dense(
            self.encoding_dim, 
            activation = 'linear', 
            name = 'bottleneck'
        )(x)
        
        # Decoder 
        x = bottleneck
        for dim in reversed(hidden_dims):
            x = Dense(dim, activation = 'relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            
        output_layer = Dense(self.input_dim, activation = 'linear')(x)
        
        # Models
        self.autoencoder = Model(
            input_layer, 
            output_layer, 
            name = 'autoencoder'
        )
        self.encoder = Model(
            input_layer,
            bottleneck, 
            name = 'encoder'
        )
        
        # Compile
        self.autoencoder.compile(
            optimizer=Adam(learning_rate = self.learning_rate),
            loss = 'mse',
            metrics = ['mae', 'mse']
        )
        
        if self.prints:
            self.autoencoder.summary()
        
    def prepare_data(
        self,
    ):
        
        self.scaled_features = self.scaler.fit_transform(self.features)
        
        self.train_data, self.test_data = train_test_split(
            self.scaled_features, 
            test_size = self.test_size,
            random_state = self.random_state,
            shuffle = False 
        )
        
        if self.prints:
            print(f"Training samples: {len(self.train_data)}")
            print(f"Test samples: {len(self.test_data)}")
            print(f"Feature shape: {self.scaled_features.shape}")
            
        return self.train_data, self.test_data
    
    def train(
        self,
    ):

        callbacks = []
        if self.use_callbacks:
            callbacks = [
                EarlyStopping(patience = 10, restore_best_weights = True),
                ReduceLROnPlateau(factor = 0.5, patience = 5),
            ]

        self.history = self.autoencoder.fit(
            self.train_data,
            self.train_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.test_data, self.test_data),
            verbose=self.verbose,
            shuffle=False,
            callbacks=callbacks
        )
        
        return self.history
    
    def add_latent_data(self):
            
        encoded_features = self.encoder.predict(self.scaled_features, verbose=0)
        
        for i in range(self.encoding_dim):
            self.new_data[f'{self.prefered_name}_{i+1}'] = encoded_features[:, i]
        
        return self.new_data
    
    def evaluate(
        self
    ):

        train_reconstruction = self.autoencoder.predict(self.train_data, verbose=0)
        test_reconstruction = self.autoencoder.predict(self.test_data, verbose=0)
        
        train_mse = np.mean(np.square(self.train_data - train_reconstruction))
        test_mse = np.mean(np.square(self.test_data - test_reconstruction))
        
        train_mae = np.mean(np.abs(self.train_data - train_reconstruction))
        test_mae = np.mean(np.abs(self.test_data - test_reconstruction))
        
        self.metrics = {
            'final_training_loss': self.history.history['loss'][-1],
            'final_validation_loss': self.history.history['val_loss'][-1],
            'training_reconstruction_mse': train_mse,
            'test_reconstruction_mse': test_mse,
            'training_reconstruction_mae': train_mae,
            'test_reconstruction_mae': test_mae,
            'generalization_gap_mse': test_mse - train_mse,
            'generalization_gap_mae': test_mae - train_mae
        }
        
        if self.prints:
            print("\n=== Model Evaluation ===")
            for key, value in self.metrics.items():
                print(f"{key}: {value:.6f}")
                
        return self.metrics
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        if 'mae' in self.history.history:
            ax2.plot(self.history.history['mae'], label='Training MAE')
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_latent_space(
        self, 
        df_with_latent: pd.DataFrame, 
        figsize: Tuple[int, int] = (12, 8)
    ):
        if self.encoding_dim < 2:
            print("Cannot plot latent space with encoding_dim < 2")
            return
            
        plt.figure(figsize = figsize)
        
        latent_1 = df_with_latent[f'{self.prefered_name}_1'].values
        latent_2 = df_with_latent[f'{self.prefered_name}_2'].values
        color_values = df_with_latent[self.target_col].values
        
        train_size = len(self.train_data)
        
        scatter1 = plt.scatter(
            latent_1[:train_size], 
            latent_2[:train_size], 
            c = color_values[:train_size], 
            cmap='viridis', 
            alpha=0.7, 
            label='Train', 
            s=20
        )
        
        scatter2 = plt.scatter(
            latent_1[train_size:],
            latent_2[train_size:], 
            c=color_values[train_size:], 
            cmap='plasma', 
            alpha=0.7, 
            marker='x',
            label='Test', 
            s=20
        )
        
        plt.colorbar(scatter1, label = self.target_col)
        plt.title(f'Latent Space Representation (Encoding dim: {self.encoding_dim})')
        plt.xlabel(f'{self.prefered_name}_1')
        plt.ylabel(f'{self.prefered_name}_2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def get_reconstruction_errors(
        self
    ):

        all_reconstructions = self.autoencoder.predict(self.scaled_features, verbose=0)
        reconstruction_errors = np.mean(np.square(self.scaled_features - all_reconstructions), axis=1)
        
        errors_df = pd.DataFrame({
            'reconstruction_error': reconstruction_errors,
            'is_test': [False] * len(self.train_data) + [True] * len(self.test_data)
        }, index=self.data.index)
        
        return errors_df
    
    def run_full_pipeline(
        self, 
        plots: bool = True
    ):
  
        # 1. Build model
        if self.prints:
            print("Building model...")
        self.model_build()
        
        # 2. Prepare data
        if self.prints:
            print("Preparing data...")
        self.prepare_data()
        
        # 3. Train model
        if self.prints:
            print("Training model...")
        self.train()
        
        # 4. Add latent features
        if self.prints:
            print("Extracting latent features...")
        data_with_latent = self.add_latent_data()
        
        # 5. Evaluate model
        if self.prints:
            print("Evaluating model...")
        metrics = self.evaluate()
        
        # 6. Get reconstruction errors
        if self.prints:
            print("Calculating reconstruction errors...")
        errors_data = self.get_reconstruction_errors()
        
        if plots:
            self.plot_training_history()
            if self.encoding_dim >= 2:
                self.plot_latent_space(data_with_latent)
                
        results = {
            'data_with_latent': data_with_latent,
            'metrics': metrics,
            'errors_data': errors_data,
            'model': self.autoencoder,
            'encoder': self.encoder,
            'history': self.history
        }
        
        return results