import itertools
import torch
import json
import pandas as pd
from models.section4.model import MatchupPredictionModel


def generate_all_matchups(team_ids, season=2025):
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


def generate_submission(team_ids, model_path, output_path='submission.csv'):
    """
    Generate predictions for all possible matchups in 2025.
    Args:
        team_ids (list): List of all team IDs.
        model_path (str): Path to the trained model.
        output_path (str): Path to save the submission file.
    """
    # Generate all possible matchups for 2025
    matchups = generate_all_matchups(team_ids, season=2025)

    print("Sample of matchups DataFrame:")
    print(matchups.head())

    print("Unique values in the ID column:")
    print(matchups['ID'].unique())

    # Extract team IDs
    matchups['WTeamID'] = matchups['ID'].apply(lambda x: int(x.split('_')[1]))
    matchups['LTeamID'] = matchups['ID'].apply(lambda x: int(x.split('_')[2]))

    # Prepare input data for the model
    X_submit = matchups[['WTeamID', 'LTeamID']].values
    X_submit = torch.tensor(X_submit, dtype=torch.float32)

    # Add a placeholder for score differences
    score_diff = torch.zeros(X_submit.shape[0], dtype=torch.float32)
    X_submit = torch.cat((X_submit, score_diff.view(-1, 1)), dim=1)

    # Load the model
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    num_teams = metadata['num_teams']
    model = MatchupPredictionModel(num_teams=num_teams)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Make predictions
    with torch.no_grad():
        predictions = model(X_submit).squeeze().numpy()

    # Add predictions to the matchups DataFrame
    matchups['Pred'] = predictions

    # Save the submission file
    matchups[['ID', 'Pred']].to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")