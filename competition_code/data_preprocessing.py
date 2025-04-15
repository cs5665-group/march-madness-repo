import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath: str) -> tuple:
    """
    Load and preprocess data with random split.
    Args:
        filepath (str): Path to the dataset file.
    Returns:
        tuple: X_train, X_test, y_train, y_test, team_ids
    """
    df = pd.read_csv(filepath)
    
    # Create a copy of the dataframe with flipped teams to simulate losses
    losing_df = df.copy()
    losing_df['WTeamID'] = df['LTeamID']
    losing_df['LTeamID'] = df['WTeamID']
    losing_df['ScoreDiff'] = -1 * (df['WScore'] - df['LScore'])
    losing_df['Result'] = 0 # Losing
    
    # Add result column to original dataframe
    df['ScoreDiff'] = df['WScore'] - df['LScore']
    df['Result'] = 1 # Winning
    
    # Combine winning and losing data
    combined_df = pd.concat([df, losing_df], ignore_index=True)

    # Features: Team IDs and score difference
    X = combined_df[['WTeamID', 'LTeamID', 'ScoreDiff']].values
    y = combined_df['Result'].values

    # Extract unique team IDs
    team_ids = pd.concat([df['WTeamID'], df['LTeamID']]).unique()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize the score difference
    scaler = StandardScaler()
    X_train[:, 2] = scaler.fit_transform(X_train[:, 2].reshape(-1, 1)).flatten()
    X_test[:, 2] = scaler.transform(X_test[:, 2].reshape(-1, 1)).flatten()

    return X_train, X_test, y_train, y_test, team_ids

def load_time_based_data(filepath: str) -> tuple:
    """
    Load and preprocess data with time-based split (more realistic for tournament predictions).
    Args:
        filepath (str): Path to the dataset file.
    Returns:
        tuple: X_train, X_val, y_train, y_val, team_ids
    """
    df = pd.read_csv(filepath)
    
    # Extract season information
    seasons = df['Season'].unique()
    seasons.sort()  # Sort seasons chronologically
    
    # Use the last season as validation set
    val_season = seasons[-1]
    
    # Split data by season
    train_df = df[df['Season'] < val_season]
    val_df = df[df['Season'] == val_season]
    
    print(f"Training on seasons {seasons[0]}-{seasons[-2]}, validating on season {val_season}")
    
    # Process training data with flipped teams for losses
    train_winning_df = train_df.copy()
    train_winning_df['ScoreDiff'] = train_winning_df['WScore'] - train_winning_df['LScore']
    train_winning_df['Result'] = 1  # Winning
    
    train_losing_df = train_df.copy()
    train_losing_df['WTeamID'], train_losing_df['LTeamID'] = train_losing_df['LTeamID'], train_losing_df['WTeamID']
    train_losing_df['ScoreDiff'] = -1 * (train_losing_df['WScore'] - train_losing_df['LScore'])
    train_losing_df['Result'] = 0  # Losing
    
    # Process validation data
    val_winning_df = val_df.copy()
    val_winning_df['ScoreDiff'] = val_winning_df['WScore'] - val_winning_df['LScore']
    val_winning_df['Result'] = 1  # Winning
    
    val_losing_df = val_df.copy()
    val_losing_df['WTeamID'], val_losing_df['LTeamID'] = val_losing_df['LTeamID'], val_losing_df['WTeamID']
    val_losing_df['ScoreDiff'] = -1 * (val_losing_df['WScore'] - val_losing_df['LScore'])
    val_losing_df['Result'] = 0  # Losing
    
    # Combine datasets
    train_combined_df = pd.concat([train_winning_df, train_losing_df], ignore_index=True)
    val_combined_df = pd.concat([val_winning_df, val_losing_df], ignore_index=True)
    
    # Extract features and labels
    X_train = train_combined_df[['WTeamID', 'LTeamID', 'ScoreDiff']].values
    y_train = train_combined_df['Result'].values
    
    X_val = val_combined_df[['WTeamID', 'LTeamID', 'ScoreDiff']].values
    y_val = val_combined_df['Result'].values
    
    # Standardize score differences
    scaler = StandardScaler()
    X_train[:, 2] = scaler.fit_transform(X_train[:, 2].reshape(-1, 1)).flatten()
    X_val[:, 2] = scaler.transform(X_val[:, 2].reshape(-1, 1)).flatten()
    
    # Get all unique team IDs
    team_ids = np.unique(np.concatenate([
        train_df['WTeamID'].unique(), 
        train_df['LTeamID'].unique(),
        val_df['WTeamID'].unique(), 
        val_df['LTeamID'].unique()
    ]))
    
    return X_train, X_val, y_train, y_val, team_ids