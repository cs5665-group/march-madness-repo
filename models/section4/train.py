import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.section4.model import MatchupPredictionModel
from models.section4.data_preprocessing import load_and_preprocess_data

def train_model(filepath: str, num_epochs: int = 10, batch_size: int = 64, learning_rate: float = 0.001): 
    
    # load and preprocess data 
    X_train, X_test, y_train, y_test, team_ids = load_and_preprocess_data(filepath)
    
    # Convert to pytorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
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
        
    # Save the model
    torch.save(model.state_dict(), 'matchup_prediction_model.pth')
    with open('model_metadata.json', 'w') as f:
        json.dump({'num_teams': num_teams}, f)
    print("Model saved as matchup_prediction_model.pth")
    