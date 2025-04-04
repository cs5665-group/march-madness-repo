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


class ModelBuilder:
    
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = None
        self.val_loader = None
    
    def set_data_loaders(self, 
                        train_loader: DataLoader, 
                        val_loader: DataLoader) -> 'ModelBuilder':
        """Set data loaders for training and validation."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        return self
    
    def create_nn_model(self, 
                       input_dim: int, 
                       hidden_dims: List[int] = [128, 64],
                       output_dim: int = 1) -> 'ModelBuilder':
        """Create a neural network model with specified architecture."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Create model
        self.model = nn.Sequential(*layers)
        self.model.to(self.device)
        
        return self
    
    def set_optimizer(self, 
                     optimizer_class: Callable = torch.optim.Adam, 
                     **optimizer_params) -> 'ModelBuilder':
        """Set optimizer for training the model."""
        if self.model is None:
            raise ValueError("Model must be created before setting optimizer")
            
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
        return self
    
    def set_loss_function(self, 
                         loss_fn: Callable = nn.MSELoss()) -> 'ModelBuilder':
        """Set loss function for training."""
        self.loss_fn = loss_fn
        return self
    
    def train_epoch(self) -> float:
        """Train the model for one epoch and return average loss."""
        if any(x is None for x in [self.model, self.optimizer, self.loss_fn, self.train_loader]):
            raise ValueError("Model, optimizer, loss function, and train_loader must be set before training")
            
        self.model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in self.train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_X).squeeze()
            loss = self.loss_fn(outputs, batch_y)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self) -> float:
        """Evaluate the model on validation data and return average loss."""
        if any(x is None for x in [self.model, self.loss_fn, self.val_loader]):
            raise ValueError("Model, loss function, and val_loader must be set before evaluation")
            
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():  # Disable gradient computation
            for batch_X, batch_y in self.val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X).squeeze()
                
                loss = self.loss_fn(outputs, batch_y)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    
    def train(self, num_epochs: int = 5, verbose: bool = True) -> Dict[str, List[float]]:
        """Train the model for specified number of epochs."""
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Validation Loss: {val_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def create_ensemble_model(self, 
                             model_class: Callable = ensemble.RandomForestClassifier, 
                             **model_params) -> 'ModelBuilder':
        """Create a scikit-learn ensemble model."""
        self.model = model_class(**model_params)
        return self
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> 'ModelBuilder':
        """Train a scikit-learn ensemble model."""
        if self.model is None:
            raise ValueError("Model must be created before training")
            
        self.model.fit(X_train, y_train)
        return self
    
    def evaluate_ensemble(self, 
                         X_val: np.ndarray, 
                         y_val: np.ndarray, 
                         metric: Callable = mean_squared_error) -> float:
        """Evaluate a scikit-learn ensemble model."""
        if self.model is None:
            raise ValueError("Model must be created before evaluation")
            
        y_pred = self.model.predict(X_val)
        return metric(y_val, y_pred)
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions using the PyTorch model."""
        if self.model is None:
            raise ValueError("Model must be created before making predictions")
            
        self.model.eval()
        X = X.to(self.device)
        
        with torch.no_grad():
            return self.model(X).squeeze()
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the scikit-learn ensemble model."""
        if self.model is None:
            raise ValueError("Model must be created before making predictions")
            
        return self.model.predict(X)