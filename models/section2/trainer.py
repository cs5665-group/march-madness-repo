import torch.optim as optim
import torch.nn as nn
from models.section2.model import GCNModel
from models.section2.ingestor import DataLoader

class Trainer:
    def __init__(self, model:GCNModel, data_loader:DataLoader, lr:float=0.01, weight_decay:float=5e-4) -> None:
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def train(self, epochs=100) -> None:
        self.model.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            output = self.model(self.data_loader.node_features, self.data_loader.edge_index)
            
            # fix the indexing of output tensor
            # Get embedings for first team 
            team1_embeddings = output[self.data_loader.train_pairs[0]]
            
            # Get embeddings for second team
            team2_embeddings = output[self.data_loader.train_pairs[1]]
            
            #calculate the logits as the difference between the embeddings
            logits = (team1_embeddings - team2_embeddings).squeeze()
            train_loss = self.criterion(logits, self.data_loader.train_labels)
            train_loss.backward()
            self.optimizer.step()
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss.item()}")
         
    def evaluate(self) -> None:
        self.model.eval()
        
        output = self.model(self.data_loader.node_features, self.data_loader.edge_index)

        # Fix the indexing of output tensor
        team1_embeddings = output[self.data_loader.val_pairs[0]]
        team2_embeddings = output[self.data_loader.val_pairs[1]]
        
        logits = (team1_embeddings - team2_embeddings).squeeze()
        val_loss = self.criterion(logits, self.data_loader.val_labels)   
        
        print(f"Validation Loss: {val_loss.item()}")