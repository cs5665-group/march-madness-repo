import torch
import torch.nn as nn

class LogRegsModel(nn.Module):
    """
    Logistic regression model for predicting match outcomes.
    """
    def __init__(self, num_teams: int, embedding_dim=16) -> None:
        super(LogRegsModel, self).__init__()
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        
        # Add BatchNorm to improve stability
        self.bn = nn.BatchNorm1d(embedding_dim * 2 + 1)
        
        # Output layer
        self.linear = nn.Linear(embedding_dim * 2 + 1, 1)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, idsTensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        team1 = self.team_embedding(idsTensor[:, 0].long())
        team2 = self.team_embedding(idsTensor[:, 1].long())
        score_diff = idsTensor[:, 2].unsqueeze(1)
        
        # Concatenate features
        features = torch.cat([team1, team2, score_diff], dim=1)
        
        # Apply normalization and dropout
        features = self.bn(features)
        features = self.dropout(features)
        
        # Linear layer
        logits = self.linear(features)
        
        return torch.sigmoid(logits)