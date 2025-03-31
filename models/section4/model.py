import torch 
import torch.nn as nn

class MatchupPredictionModel(nn.Module):
    
    def __init__(self, num_teams: int, emedding_dim=32):
        """
        Neural network model for predicting match outcomes.
        Args:
            num_teams (int): Number of teams in the dataset.
            embedding_dim (int): Dimension of the embedding layer.
        """
        
        super(MatchupPredictionModel, self).__init__()
        self.team_embedding = nn.Embedding(num_teams, emedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(emedding_dim * 2 + 1, 64), # 2 embeddings + score difference
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Output between 0 and 1
        )
        
        
    # Forward pass 
    def forward(self, idsTensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor containing team IDs and score difference.
            Returns:
                torch.Tensor: Predicted probabilities of winning.
        """
        
        team1 = self.team_embedding(idsTensor[:, 0].long())
        team2 = self.team_embedding(idsTensor[:, 1].long())
        score_diff = idsTensor[:, 2].unsqueeze(1)
        features = torch.cat([team1, team2, score_diff], dim=1)
        return self.fc(features)