import numpy as np
import pandas as pd
import optuna 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple, Union, Callable

class ProcessingBuilder:
    
    def __init__(self):
        self.imputer = None
        self.scaler = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.X_test = None
        self.feature_columns = None
        self.target_column = None
    
    def set_feature_columns(self, feature_columns:List[str]) -> 'ProcessingBuilder':
        """
            Set the feature columns to use for training and inference
            
            Args:
                feature_columns: List[str]: List of feature columns
        """
        self.feature_columns = feature_columns
        return self
    
    def set_target_column(self, target_column:str) -> 'ProcessingBuilder':
        """
            Set the target column to use for training and inference
            
            Args:
                target_column: str: Target column
        """
        self.target_column = target_column
        return self
    
    def prepare_data(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None, 
                     val_size: float = 0.2, random_state: int = 42) -> 'ProcessingBuilder':
        if self.feature_columns is None or self.target_column is None:
            raise ValueError("Feature columns and target column must be set before preparing data")
        
        X = train_df[self.feature_columns].fillna(-1)
        y = train_df[self.target_column]
        
        # Split data into train and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=val_size, random_state=random_state
        )
        
        # Handle test data if provided
        if test_df is not None:
            self.X_test = test_df[self.feature_columns].fillna(-1)
        
        return self
    
    def apply_imputation(self, strategy: str = 'mean') -> 'ProcessingBuilder':
        """Apply imputation to handle missing values."""
        self.imputer = SimpleImputer(strategy=strategy)
        
        # Fit and transform training data
        self.X_train = self.imputer.fit_transform(self.X_train)
        
        # Transform validation and test data
        if self.X_val is not None:
            self.X_val = self.imputer.transform(self.X_val)
        
        if self.X_test is not None:
            self.X_test = self.imputer.transform(self.X_test)
        
        return self
    
    def apply_scaling(self) -> 'ProcessingBuilder':
        """Apply scaling to normalize feature values."""
        self.scaler = StandardScaler()
        
        # Fit and transform training data
        self.X_train = self.scaler.fit_transform(self.X_train)
        
        # Transform validation and test data
        if self.X_val is not None:
            self.X_val = self.scaler.transform(self.X_val)
        
        if self.X_test is not None:
            self.X_test = self.scaler.transform(self.X_test)
        
        return self
    
    def get_data(self) -> Dict[str, np.ndarray]:
        """Get preprocessed data as numpy arrays."""
        result = {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_val': self.X_val,
            'y_val': self.y_val
        }
        
        if self.X_test is not None:
            result['X_test'] = self.X_test
            
        return result
    
    
    def create_torch_datasets(self, batch_size: int = 64) -> Dict[str, Union[TensorDataset, DataLoader]]:
        """Create PyTorch datasets and dataloaders from preprocessed data."""
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train.values, dtype=torch.float32)
        
        X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(self.y_val.values, dtype=torch.float32)
        
        # Create TensorDatasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'train_loader': train_loader,
            'val_loader': val_loader
        }
        