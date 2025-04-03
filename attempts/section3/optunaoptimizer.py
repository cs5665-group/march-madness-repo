import optuna 
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn import ensemble
from sklearn.metrics import log_loss
from typing import Dict, Optional, Any, List
from models.section3.processingbuilder import ProcessingBuilder
from models.section3.modelbuilder import ModelBuilder


class OptunaOptimizer:
    """Class for hyperparameter optimization using Optuna."""
    
    def __init__(self, 
                preprocessor: ProcessingBuilder,
                n_trials: int = 100,
                timeout: Optional[int] = None):
        self.preprocessor = preprocessor
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = None
        self.best_params = None
        
    def nn_objective(self, trial) -> float:
        """Objective function for neural network hyperparameter tuning."""
        # Sample hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        hidden_dims = [
            trial.suggest_int("hidden_dim_1", 32, 256),
            trial.suggest_int("hidden_dim_2", 16, 128)
        ]
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        
        # Get data dimensions
        input_dim = self.preprocessor.X_train.shape[1]
        
        # Create datasets and dataloaders
        datasets = self.preprocessor.create_torch_datasets(batch_size=batch_size)
        
        # Build and train model
        model_builder = ModelBuilder()
        model_builder.set_data_loaders(datasets['train_loader'], datasets['val_loader']) \
                    .create_nn_model(input_dim, hidden_dims) \
                    .set_optimizer(torch.optim.Adam, lr=lr) \
                    .set_loss_function(nn.MSELoss())
        
        # Train for a few epochs
        history = model_builder.train(num_epochs=3, verbose=False)
        
        # Return final validation loss
        return history['val_losses'][-1]
    
    def ensemble_objective(self, trial) -> float:
        """Objective function for ensemble model hyperparameter tuning."""
        # Sample hyperparameters for RandomForest
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        
        # Build and train model
        model_builder = ModelBuilder()
        model_builder.create_ensemble_model(
            ensemble.RandomForestClassifier,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        
        # Train model
        model_builder.train_ensemble(self.preprocessor.X_train, self.preprocessor.y_train)
        
        # Evaluate model
        val_score = model_builder.evaluate_ensemble(
            self.preprocessor.X_val, 
            self.preprocessor.y_val,
            log_loss
        )
        
        return val_score
    
    def optimize_nn(self) -> Dict[str, Any]:
        """Run hyperparameter optimization for neural network."""
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self.nn_objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        return {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'study': self.study
        }
    
    def optimize_ensemble(self) -> Dict[str, Any]:
        """Run hyperparameter optimization for ensemble model."""
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self.ensemble_objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        return {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'study': self.study
        }
    
    def create_best_model(self, model_type: str = 'nn') -> ModelBuilder:
        """Create a model with the best hyperparameters."""
        if self.best_params is None:
            raise ValueError("Must run optimization before creating best model")
        
        if model_type == 'nn':
            # Get data dimensions
            input_dim = self.preprocessor.X_train.shape[1]
            
            # Create datasets and dataloaders
            datasets = self.preprocessor.create_torch_datasets(
                batch_size=self.best_params.get('batch_size', 64)
            )
            
            # Build model with best parameters
            model_builder = ModelBuilder()
            model_builder.set_data_loaders(datasets['train_loader'], datasets['val_loader']) \
                        .create_nn_model(input_dim, [
                            self.best_params.get('hidden_dim_1', 128), 
                            self.best_params.get('hidden_dim_2', 64)
                        ]) \
                        .set_optimizer(torch.optim.Adam, lr=self.best_params.get('lr', 0.001)) \
                        .set_loss_function(nn.MSELoss())
        
        elif model_type == 'ensemble':
            # Build ensemble model with best parameters
            model_builder = ModelBuilder()
            model_builder.create_ensemble_model(
                ensemble.RandomForestClassifier,
                n_estimators=self.best_params.get('n_estimators', 100),
                max_depth=self.best_params.get('max_depth', 10),
                min_samples_split=self.best_params.get('min_samples_split', 2),
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model_builder
    
    def generate_submission(self, 
                           model_builder: ModelBuilder, 
                           submission_df: pd.DataFrame,
                           feature_columns: List[str],
                           output_path: str = 'submission.csv') -> pd.DataFrame:
        """
        Generate predictions for submission data and save to CSV.
        
        Args:
            model_builder: Trained ModelBuilder instance with model
            submission_df: DataFrame containing submission data
            feature_columns: List of feature columns to use for prediction
            output_path: Path to save the submission CSV
            
        Returns:
            DataFrame with predictions
        """
        if model_builder.model is None:
            raise ValueError("Model must be trained before generating submission")
            
        # Prepare submission features
        X_submit = submission_df[feature_columns].fillna(-1)
        
        # Apply the same preprocessing steps used during training
        if hasattr(self.preprocessor, 'imputer') and self.preprocessor.imputer is not None:
            X_submit_imputed = self.preprocessor.imputer.transform(X_submit)
        else:
            X_submit_imputed = X_submit
            
        if hasattr(self.preprocessor, 'scaler') and self.preprocessor.scaler is not None:
            X_submit_scaled = self.preprocessor.scaler.transform(X_submit_imputed)
        else:
            X_submit_scaled = X_submit_imputed
        
        # Generate predictions based on model type
        if isinstance(model_builder.model, torch.nn.Module):
            # Convert to PyTorch tensor
            X_submit_tensor = torch.tensor(X_submit_scaled, dtype=torch.float32)
            
            # Move to appropriate device
            device = model_builder.device
            X_submit_tensor = X_submit_tensor.to(device)
            
            # Set model to evaluation mode
            model_builder.model.eval()
            
            # Make predictions
            with torch.no_grad():
                y_preds = model_builder.model(X_submit_tensor).cpu().numpy()
                
            # Reshape if needed
            if len(y_preds.shape) > 1 and y_preds.shape[1] == 1:
                y_preds = y_preds.flatten()
                
        else:
            # For scikit-learn models
            y_preds = model_builder.model.predict_proba(X_submit_scaled)[:, 1]
        
        # Create submission DataFrame
        submission = submission_df.copy()
        submission['Pred'] = y_preds
        
        # Save to CSV
        submission[['ID', 'Pred']].to_csv(output_path, index=False)
        
        return submission