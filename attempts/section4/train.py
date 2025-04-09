import torch
import json
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from attempts.section4.data_preprocessing import load_time_based_data
from attempts.section4.models.neural_network_model import MatchupPredictionModel
from attempts.section4.models.log_regs import LogRegsModel
from attempts.section4.models.binary_class import BinaryClassificationModel
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss
from attempts.section4.loss_tracker import LossTracker
from attempts.section4.plotting import evaluate_and_visualize_model

class BrierLoss(nn.Module):
    """Custom Brier Score Loss function"""
    def __init__(self):
        super(BrierLoss, self).__init__()
    
    def forward(self, input, target):
        return torch.mean((input - target) ** 2)

def train_nn_model(filepath: str, num_epochs: int = 30, batch_size: int = 64, learning_rate: float = 0.001, dropout_rate: float = 0.7): 
    """
    Train a neural network model with anti-overfitting techniques
    """
    # Always use time-based data split for more realistic evaluation
    X_train, X_val, y_train, y_val, _ = load_time_based_data(filepath)
    print("Using time-based data split for validation")
    
    # Convert to pytorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model with smaller embedding
    num_teams = int(max(X_train[:, 0].max(), X_train[:, 1].max() + 1))
    embedding_dim = 16  # Reduced embedding dimension to prevent overfitting
    model = MatchupPredictionModel(num_teams=num_teams, embedding_dim=embedding_dim, dropout_rate=dropout_rate)
    
    # Use Brier loss for better calibration
    criterion = BrierLoss()
    
    # Use AdamW with strong weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    # More aggressive learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )
    
    # Initialize loss tracker 
    loss_tracker = LossTracker()
        
    # Early stopping parameters
    best_loss = float('inf')
    best_model_state = None
    patience = 7  # Reduced patience
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch.squeeze())
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
            
        # Validation phase
        model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch).squeeze()
                loss = criterion(predictions, y_batch.squeeze())
                val_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Track losses for visualization
        loss_tracker.update(train_loss, val_loss)
        
        # Apply learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (best)")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model state
    model.load_state_dict(best_model_state)
    
    os.makedirs('visualizations', exist_ok=True)
    
    loss_tracker.plot("Neural Network", save_pth='visualizations/nn_training_history.png')
    
    # Concatenate all predictions and targets for final evaluation
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    brier_score = brier_score_loss(all_targets, all_predictions)
    binary_predictions = [1 if p >= 0.5 else 0 for p in all_predictions]
    accuracy = accuracy_score(all_targets, binary_predictions)
    logloss = log_loss(all_targets, all_predictions)
    
    print(f"Final Brier Score: {brier_score:.4f}")
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Final Log Loss: {logloss:.4f}")
    
    # Generate all evaluation plots
    evaluate_and_visualize_model(model, X_val, y_val, 'Neural_Network', 'visualizations')
    
    # Save the model with proper metadata
    torch.save(model.state_dict(), 'neural_net_model.pth')
    with open('model_metadata.json', 'w') as f:
        json.dump({
            'num_teams': int(num_teams),
            'embedding_dim': embedding_dim,
            'model_type': 'neural_network'
        }, f)
    print("Model saved as neural_net_model.pth")
    
    return model, brier_score, accuracy

def train_log_reg_model(filepath: str, num_epochs: int = 30, batch_size: int = 64, learning_rate: float = 0.001):
    """
    Train a logistic regression model with enhanced techniques
    """
    # Use time-based data split
    X_train, X_val, y_train, y_val, _ = load_time_based_data(filepath)
    print("Using time-based data split for validation")

    # Convert to pytorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model with batch normalization
    num_teams = int(max(X_train[:, 0].max(), X_train[:, 1].max() + 1))
    embedding_dim = 16  # Reduced from 64
    model = LogRegsModel(num_teams=num_teams, embedding_dim=embedding_dim)
    
    # Use Brier loss
    criterion = BrierLoss()
    
    # Add weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    loss_tracker = LossTracker()
    
    # Early stopping parameters
    best_loss = float('inf')
    best_model_state = None
    patience = 7
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch.squeeze())
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch).squeeze()
                loss = criterion(predictions, y_batch.squeeze())
                val_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Track losses for visualization
        loss_tracker.update(train_loss, val_loss)
        
        # Apply learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (best)")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Plot training history
    loss_tracker.plot("Logistic Regression", save_pth='visualizations/log_reg_training_history.png')
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    brier_score = brier_score_loss(all_targets, all_predictions)
    binary_predictions = [1 if p >= 0.5 else 0 for p in all_predictions]
    accuracy = accuracy_score(all_targets, binary_predictions)
    logloss = log_loss(all_targets, all_predictions)
    
    print(f"Final Brier Score: {brier_score:.4f}")
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Final Log Loss: {logloss:.4f}")
    
    # Generate all evaluation plots
    evaluate_and_visualize_model(model, X_val, y_val, 'Logistic_Regression', 'visualizations')
    
    # Save the model with metadata
    torch.save(model.state_dict(), 'log_reg_model.pth')
    with open('model_metadata.json', 'w') as f:
        json.dump({
            'num_teams': int(num_teams),
            'embedding_dim': embedding_dim,
            'model_type': 'log_reg'
        }, f)
    print("Model saved as log_reg_model.pth")
    
    return model, brier_score, accuracy

def train_binary_classification_model(filepath:str, num_epochs: int = 30, batch_size: int = 64, learning_rate: float = 0.001): 
    """
    Train a binary classification model with improved techniques
    """
    # Use time-based data split
    X_train, X_val, y_train, y_val, _ = load_time_based_data(filepath)
    print("Using time-based data split for validation")
    
    # Convert to pytorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    num_teams = int(max(X_train[:, 0].max(), X_train[:, 1].max() + 1))
    embedding_dim = 16  # Reduced from 64
    model = BinaryClassificationModel(num_teams=num_teams, embedding_dim=embedding_dim)
    
    # Use Brier loss
    criterion = BrierLoss()
    
    # Add weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Initialize loss tracker
    loss_tracker = LossTracker()
    
    # Early stopping parameters
    best_loss = float('inf')
    best_model_state = None
    patience = 7
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch.squeeze())
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch).squeeze()
                loss = criterion(predictions, y_batch.squeeze())
                val_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Track losses for visualization
        loss_tracker.update(train_loss, val_loss)
        
        # Apply learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (best)")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Plot training history
    loss_tracker.plot("Binary Classification", save_pth='visualizations/binary_class_training_history.png')
        
    # Final evaluation
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Calculate metrics
    brier_score = brier_score_loss(all_targets, all_predictions)
    binary_predictions = [1 if p >= 0.5 else 0 for p in all_predictions]
    accuracy = accuracy_score(all_targets, binary_predictions)
    logloss = log_loss(all_targets, all_predictions)
    
    print(f"Final Brier Score: {brier_score:.4f}")
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Final Log Loss: {logloss:.4f}")
    
    # Generate all evaluation plots
    evaluate_and_visualize_model(model, X_val, y_val, 'Binary_Classification', 'visualizations')
    
    # Save the model with metadata
    torch.save(model.state_dict(), 'binary_class_model.pth')
    with open('model_metadata.json', 'w') as f:
        json.dump({
            'num_teams': int(num_teams),
            'embedding_dim': embedding_dim,
            'model_type': 'binary_classification'
        }, f)
    print("Model saved as binary_class_model.pth")
    
    return model, brier_score, accuracy