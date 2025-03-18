from models.section2.ingestor import DataLoader
from models.section2.model import GCNModel
from models.section2.trainer import Trainer

import os
import torch
import pandas as pd


def create_submission_file(model:GCNModel, data_loader:DataLoader) -> pd.DataFrame:
    """
        Create a submission file using the trained model and the data loader
        
        Args: 
            model: GCNModel: Trained model
            data_loader: DataLoader: DataLoader object
    """
    model.eval()
    
    # Read the template file that contains all the matchups we need to predict
    sample_submission = pd.read_csv('datasets/SampleSubmissionStage1.csv')
    print(f"Sample submission has {len(sample_submission)} rows")
    
    # Create a dictionary to store predictions
    predictions_dict = {}
    
    with torch.no_grad():
        node_embeddings = model(data_loader.node_features, data_loader.edge_index)
        
        for idx, row in sample_submission.iterrows():
            game_id = row['ID']
            
            # Skip if we've already predicted this game
            if game_id in predictions_dict:
                continue
                
            parts = game_id.split('_')
            season = int(parts[0])
            team1_id = int(parts[1])
            team2_id = int(parts[2])
            
            # check if both teams are in our dataset
            if team1_id in data_loader.team_id_to_idx and team2_id in data_loader.team_id_to_idx:
                # get team indices
                team1_idx = data_loader.team_id_to_idx[team1_id]
                team2_idx = data_loader.team_id_to_idx[team2_id]
                
                # get team embeddings
                team1_embedding = node_embeddings[team1_idx]
                team2_embedding = node_embeddings[team2_idx]
                
                # Calculate probability of team1 winning
                probability = torch.sigmoid(team1_embedding - team2_embedding).item()
                predictions_dict[game_id] = probability
                
            else:
                # if team not in our dataset, use a default prediction (0.5)
                predictions_dict[game_id] = 0.5
    
    # Create a new dataframe with the same IDs as the sample submission
    submission_df = pd.DataFrame({
        'ID': sample_submission['ID'],
        'Pred': sample_submission['ID'].map(predictions_dict)
    })
    
    # Fill any missing predictions with 0.5
    submission_df['Pred'] = submission_df['Pred'].fillna(0.5)
    
    print(f"Submission has {len(submission_df)} rows")
    
    # Save to CSV 
    submission_df.to_csv('submission.csv', index=False)
    print("Submission successfully created")
    
    return submission_df

if __name__ == "__main__":
    data_loader:DataLoader = DataLoader()
    model:GCNModel = GCNModel(num_features=data_loader.node_features.shape[1], hidden_dim=50, num_classes=1)
    trainer:Trainer = Trainer(model, data_loader)
    trainer.train(epochs=100)
    trainer.evaluate()
    
    # Create submission file
    create_submission_file(model, data_loader)