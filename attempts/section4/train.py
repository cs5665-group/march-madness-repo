import torch
import json
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from attempts.section4.data_preprocessing import load_and_preprocess_data
from attempts.section4.models.neural_network_model import MatchupPredictionModel
from attempts.section4.models.log_regs import LogRegsModel
from attempts.section4.models.binary_class import BinaryClassificationModel
from sklearn.metrics import brier_score_loss, accuracy_score
from attempts.section4.loss_tracker import LossTracker
from attempts.section4.plotting import evaluate_and_visualize_model

def train_nn_model(filepath: str, num_epochs: int = 10, batch_size: int = 64, learning_rate: float = 0.001): 
    
    # load and preprocess data 
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(filepath)
    
    # Convert to pytorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    num_teams = int(max(X_train[:, 0].max(), X_train[:, 1].max() + 1))
    model = MatchupPredictionModel(num_teams=num_teams)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Initialize loss tracker 
    loss_tracker = LossTracker()
        
    all_predictions = []
    all_targets = []
    test_loss = 0
    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )    
    
    # track best model
    best_loss = float('inf')
    best_model_state = None
    
    model.train()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch.squeeze())
            loss.backward()
            
            # add gradient clipping to preventexploring gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                predictions = model(X_batch).squeeze()
                loss = criterion(predictions, y_batch.squeeze())
                val_loss += loss.item()
                all_predictions.append(predictions.numpy())
                all_targets.append(y_batch.numpy())
                test_loss += loss.item()
        
        val_loss /= len(test_loader)
        
        loss_tracker.update(train_loss, val_loss)
        # Apply learning rate scheduling
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    
    # Load the best model state
    model.load_state_dict(best_model_state)
    
    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)
    
    loss_tracker.plot("Neural Network", save_pth='visualizations/nn_training_history.png')
    
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    brier_score = brier_score_loss(all_targets, all_predictions)
    binary_predictions = [1 if p >= 0.5 else 0 for p in all_predictions]
    accuracy = accuracy_score(all_targets, binary_predictions)
    
    print(f"Brier Score: {brier_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Generate all evaluation plots
    evaluate_and_visualize_model(model, X_test, y_test, 'Neural Network', 'visualizations')
    
    # Save the model
    torch.save(model.state_dict(), 'neural_net_model.pth')
    with open('model_metadata.json', 'w') as f:
        json.dump({'num_teams': num_teams}, f)
    print("Model saved as log_reg_model.pth")
    
    
def train_log_reg_model(filepath: str, num_epochs: int = 10, batch_size: int = 64, learning_rate: float = 0.001):
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(filepath)

    # Convert to pytorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    num_teams = int(max(X_train[:, 0].max(), X_train[:, 1].max() + 1))
    model = LogRegsModel(num_teams=num_teams)
    criterion = nn.BCELoss()  # Changed to BCELoss for binary classification with sigmoid
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_tracker = LossTracker()
    
    # Track training and validation losses
    train_losses = []
    val_losses = []
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validate after each epoch
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                predictions = model(X_batch).squeeze()
                loss = criterion(predictions, y_batch.squeeze())
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        loss_tracker.update(train_loss, val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Plot training history
    loss_tracker.plot("Logistic Regression", save_pth='visualizations/log_reg_training_history.png')
    
    # Evaluate on test data
    all_predictions = []
    all_targets = []
    test_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch.squeeze())
            test_loss += loss.item()
            
            # Store predictions and targets for further analysis
            all_predictions.append(predictions.numpy())
            all_targets.append(y_batch.numpy())

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    brier_score = brier_score_loss(all_targets, all_predictions)
    binary_predictions = [1 if p >= 0.5 else 0 for p in all_predictions]
    accuracy = accuracy_score(all_targets, binary_predictions)
    
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Generate all evaluation plots
    evaluate_and_visualize_model(model, X_test, y_test, 'Logistic Regression', 'visualizations')
    
    # Save the model
    torch.save(model.state_dict(), 'log_reg_model.pth')
    with open('model_metadata.json', 'w') as f:
        json.dump({'num_teams': num_teams}, f)
    print("Model saved as log_reg_model.pth")
    
    
def train_binary_classification_model(filepath:str, num_epochs: int = 10, batch_size: int = 64, learning_rate: float = 0.001): 
    """
    Train a binary classification model using the provided dataset.
    Args:
        filepath (str): Path to the dataset file.
        num_epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
    """
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(filepath)
    
    # Convert to pytorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    num_teams = int(max(X_train[:, 0].max(), X_train[:, 1].max() + 1))
    model = BinaryClassificationModel(num_teams=num_teams)
    criterion = nn.BCELoss()  # Changed to BCELoss for binary classification with sigmoid
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize loss tracker
    loss_tracker = LossTracker()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Validate after each epoch
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                predictions = model(X_batch).squeeze()
                loss = criterion(predictions, y_batch.squeeze())
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        
        loss_tracker.update(train_loss, val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Plot training history
    loss_tracker.plot("Binary Classification", save_pth='visualizations/binary_class_training_history.png')
        
    # Evaluate on test data
    all_predictions = []
    all_targets = []
    test_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch.squeeze())
            test_loss += loss.item()
            
            # Store predictions and targets for further analysis
            all_predictions.append(predictions.numpy())
            all_targets.append(y_batch.numpy())
            
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    brier_score = brier_score_loss(all_targets, all_predictions)
    binary_predictions = [1 if p >= 0.5 else 0 for p in all_predictions]
    accuracy = accuracy_score(all_targets, binary_predictions)
    
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Generate all evaluation plots
    evaluate_and_visualize_model(model, X_test, y_test, 'Binary Classification', 'visualizations')
    
    # Save the model
    torch.save(model.state_dict(), 'binary_class_model.pth')
    with open('model_metadata.json', 'w') as f:
        json.dump({'num_teams': num_teams}, f)
    print("Model saved as binary_class_model.pth")