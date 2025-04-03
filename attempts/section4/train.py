import torch
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from attempts.section4.data_preprocessing import load_and_preprocess_data
from attempts.section4.models.neural_network_model import MatchupPredictionModel
from attempts.section4.models.log_regs import LogRegsModel
from sklearn.metrics import brier_score_loss, accuracy_score

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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
    # Evaluate on test data
    model.eval()
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
    
    # Flatten lists of arrays into single arrays
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
        
    brier_score = brier_score_loss(all_targets, all_predictions)
    binary_predictions = [1 if p >= 0.5 else 0 for p in all_predictions]
    accuracy = accuracy_score(all_targets, binary_predictions)
    
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), 'neural_net_model.pth')
    with open('model_metadata.json', 'w') as f:
        json.dump({'num_teams': num_teams}, f)
    print("Model saved as matchup_prediction_model.pth")
    
    
def train_log_reg_model(filepath: str, num_epochs: int = 10, batch_size: int = 64, learning_rate: float = 0.001):
    
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
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
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
    
    # Save the model
    torch.save(model.state_dict(), 'log_reg_model.pth')
    with open('model_metadata.json', 'w') as f:
        json.dump({'num_teams': num_teams}, f)
    print("Model saved as log_reg_model.pth")
    
    