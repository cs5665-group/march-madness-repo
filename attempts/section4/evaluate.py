import itertools
import torch
import json
import pandas as pd
import os
from attempts.section4.models.neural_network_model import MatchupPredictionModel
from attempts.section4.models.log_regs import LogRegsModel
from attempts.section4.models.binary_class import BinaryClassificationModel

def generate_all_matchups(team_ids, season=2025) -> pd.DataFrame:
    """
    Generate all possible matchups for a given season.
    Args:
        team_ids (list): List of all team IDs.
        season (int): The season year (e.g., 2025).
    Returns:
        pd.DataFrame: DataFrame containing all possible matchups with IDs.
    """
    matchups = []
    for team1, team2 in itertools.combinations(sorted(team_ids), 2):
        matchups.append(f"{season}_{team1}_{team2}")
    return pd.DataFrame({'ID': matchups})

def generate_nerual_net_sub(teams_ids, neural_net_path, output_path='neural_net_submission.csv') -> None:
    """
    Generate predictions for all possible matchups using the neural network model.
    Args:
        teams_ids (list): List of all team IDs.
        neural_net_path (str): Path to the trained neural network model.
        output_path (str): Path to save the submission file.
    """
    # Generate all possible matchups for 2025
    matchups = generate_all_matchups(teams_ids, season=2025)

    # Extract team IDs
    matchups['WTeamID'] = matchups['ID'].apply(lambda x: int(x.split('_')[1]))
    matchups['LTeamID'] = matchups['ID'].apply(lambda x: int(x.split('_')[2]))

    # Prepare input data for the model - ONLY team IDs, no score difference for prediction
    X_submit = matchups[['WTeamID', 'LTeamID']].values
    X_submit = torch.tensor(X_submit, dtype=torch.float32)

    # Add a placeholder score difference of 0 for prediction
    score_diff = torch.zeros(X_submit.shape[0], dtype=torch.float32)
    X_submit = torch.cat((X_submit, score_diff.view(-1, 1)), dim=1)

    # Load metadata with embedding dimension
    if os.path.exists('model_metadata.json'):
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
            num_teams = metadata['num_teams']
            embedding_dim = metadata.get('embedding_dim', 16)  # Default to 16
    else:
        # Fallback values if metadata file doesn't exist
        num_teams = max(teams_ids) + 1
        embedding_dim = 16
    
    # Initialize model with correct embedding dimension
    neural_net_model = MatchupPredictionModel(num_teams=num_teams, embedding_dim=embedding_dim, dropout_rate=0.7)
    
    # Load model weights safely
    neural_net_model.load_state_dict(torch.load(neural_net_path, weights_only=True))
    neural_net_model.eval()

    # Make predictions
    with torch.no_grad():
        predictions = neural_net_model(X_submit).squeeze().numpy()

    # Add predictions to the matchups DataFrame
    matchups['Pred'] = predictions
    # Save the submission file
    matchups[['ID', 'Pred']].to_csv(output_path, index=False)
    print(f"Neural network submission file saved to {output_path}")
    
def generate_log_regs_sub(teams_ids, log_reg_path, output_path='log_reg_submission.csv') -> None:
    """
    Generate predictions for all possible matchups using the logistic regression model.
    """
    # Generate all possible matchups for 2025
    matchups = generate_all_matchups(teams_ids, season=2025)

    # Extract team IDs
    matchups['WTeamID'] = matchups['ID'].apply(lambda x: int(x.split('_')[1]))
    matchups['LTeamID'] = matchups['ID'].apply(lambda x: int(x.split('_')[2]))

    # Prepare input data - ONLY team IDs
    X_submit = matchups[['WTeamID', 'LTeamID']].values
    X_submit = torch.tensor(X_submit, dtype=torch.float32)

    # Add a placeholder score difference
    score_diff = torch.zeros(X_submit.shape[0], dtype=torch.float32)
    X_submit = torch.cat((X_submit, score_diff.view(-1, 1)), dim=1)

    # Load metadata
    if os.path.exists('model_metadata.json'):
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
            num_teams = metadata['num_teams']
            embedding_dim = metadata.get('embedding_dim', 16)
    else:
        num_teams = max(teams_ids) + 1
        embedding_dim = 16
    
    # Initialize model with embedding dimension
    log_reg_model = LogRegsModel(num_teams=num_teams, embedding_dim=embedding_dim)
    
    # Load model weights safely
    log_reg_model.load_state_dict(torch.load(log_reg_path, weights_only=True))
    log_reg_model.eval()

    # Make predictions
    with torch.no_grad():
        predictions = log_reg_model(X_submit).squeeze().numpy()

    # Add predictions to the matchups DataFrame
    matchups['Pred'] = predictions
    # Save the submission file
    matchups[['ID', 'Pred']].to_csv(output_path, index=False)
    print(f"Logistic regression submission file saved to {output_path}")
    
def generate_binnary_class_sub(teams_ids, binary_class_path, output_path='binary_class_submission.csv') -> None:
    """
    Generate predictions for all possible matchups using the binary classification model.
    """
    # Generate all possible matchups for 2025
    matchups = generate_all_matchups(teams_ids, season=2025)

    # Extract team IDs
    matchups['WTeamID'] = matchups['ID'].apply(lambda x: int(x.split('_')[1]))
    matchups['LTeamID'] = matchups['ID'].apply(lambda x: int(x.split('_')[2]))

    # Prepare input data - ONLY team IDs
    X_submit = matchups[['WTeamID', 'LTeamID']].values
    X_submit = torch.tensor(X_submit, dtype=torch.float32)

    # Add a placeholder score difference
    score_diff = torch.zeros(X_submit.shape[0], dtype=torch.float32)
    X_submit = torch.cat((X_submit, score_diff.view(-1, 1)), dim=1)

    # Load metadata
    if os.path.exists('model_metadata.json'):
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
            num_teams = metadata['num_teams']
            embedding_dim = metadata.get('embedding_dim', 16)
    else:
        num_teams = max(teams_ids) + 1
        embedding_dim = 16
    
    # Initialize model with embedding dimension
    binary_class_model = BinaryClassificationModel(num_teams=num_teams, embedding_dim=embedding_dim)
    
    # Load model weights safely
    binary_class_model.load_state_dict(torch.load(binary_class_path, weights_only=True))
    binary_class_model.eval()

    # Make predictions
    with torch.no_grad():
        predictions = binary_class_model(X_submit).squeeze().numpy()

    # Add predictions to the matchups DataFrame
    matchups['Pred'] = predictions
    # Save the submission file
    matchups[['ID', 'Pred']].to_csv(output_path, index=False)
    print(f"Binary classification submission file saved to {output_path}")