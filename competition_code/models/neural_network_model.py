import torch 
import torch.nn as nn

class MatchupPredictionModel(nn.Module):
    
    def __init__(self, num_teams: int, embedding_dim=32, dropout_rate=0.4) -> None:
        """
        Neural network model for predicting match outcomes.
        Args:
            num_teams (int): Number of teams in the dataset.
            embedding_dim (int): Dimension of the embedding layer.
        """
        
        super(MatchupPredictionModel, self).__init__()
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        
        self.layer = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.__initialize_weights()
        
    def __initialize_weights(self) -> None: 
        """
        Initialize weights of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        
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
        
        x = self.layer(features)
        
        return self.output_layer(x)