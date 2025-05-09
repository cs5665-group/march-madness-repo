import torch 
import torch.nn as nn

class BinaryClassificationModel(nn.Module):
    def __init__(self, num_teams: int, embedding_dim=16) -> None:
        super(BinaryClassificationModel, self).__init__()
        
        # Team embeddings
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        
        # Add BatchNorm and dropout
        self.bn = nn.BatchNorm1d(embedding_dim * 2 + 1)
        self.dropout = nn.Dropout(0.5)
        
        # Output layer
        self.linear = nn.Linear(embedding_dim * 2 + 1, 1)
        
    def forward(self, idsTensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            idsTensor (torch.Tensor): Input tensor containing team IDs and score difference.
        Returns:
            torch.Tensor: Predicted probabilities of winning.
        """
        team1 = self.team_embedding(idsTensor[:, 0].long())
        team2 = self.team_embedding(idsTensor[:, 1].long())
        score_diff = idsTensor[:, 2].unsqueeze(1)
        
        features = torch.cat([team1, team2, score_diff], dim=1)
        features = self.bn(features)
        features = self.dropout(features)
        
        logits = self.linear(features)
        
        return torch.sigmoid(logits)