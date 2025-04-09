import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(filepath: str) -> tuple:
    """
    Load and preprocess the NCAA tournament data.
    Args:
        filepath (str): Path to the dataset file.
    Returns:
        X_train, X_test, y_train, y_test, team_ids: Processed training and testing data and unique team IDs.
    """
    df = pd.read_csv(filepath)

    # Calculate score difference and assign results
    df['ScoreDiff'] = df['WScore'] - df['LScore']
    df['Result'] = 1  # Winning team is labeled as 1

    # Create a losing team version of the data
    losing_df = df.copy()
    losing_df['WTeamID'], losing_df['LTeamID'] = losing_df['LTeamID'], losing_df['WTeamID']
    losing_df['ScoreDiff'] = -losing_df['ScoreDiff']
    losing_df['Result'] = 0  # Losing team is labeled as 0

    # Combine winning and losing data
    combined_df = pd.concat([df, losing_df], ignore_index=True)

    # Features: Team IDs and score difference
    X = combined_df[['WTeamID', 'LTeamID', 'ScoreDiff']].values
    y = combined_df['Result'].values

    # Extract unique team IDs
    team_ids = pd.concat([df['WTeamID'], df['LTeamID']]).unique()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the score difference
    scaler = StandardScaler()
    X_train[:, 2] = scaler.fit_transform(X_train[:, 2].reshape(-1, 1)).flatten()
    X_test[:, 2] = scaler.transform(X_test[:, 2].reshape(-1, 1)).flatten()

    return X_train, X_test, y_train, y_test, team_ids

def load_time_based_data(filepath: str):
    """
    Load and preprocess data with time-based split (more realistic for tournament predictions)
    
    Args:
        filepath (str): Path to the dataset file
    
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
    
    # Process training data
    X_train_data = []
    y_train_data = []
    
    for _, row in train_df.iterrows():
        # Team 1 wins
        X_train_data.append([row['WTeamID'], row['LTeamID'], row['WScore'] - row['LScore']])
        y_train_data.append(1)
        
        # Team 2 wins (symmetric case)
        X_train_data.append([row['LTeamID'], row['WTeamID'], row['LScore'] - row['WScore']])
        y_train_data.append(0)
    
    # Process validation data
    X_val_data = []
    y_val_data = []
    
    for _, row in val_df.iterrows():
        X_val_data.append([row['WTeamID'], row['LTeamID'], row['WScore'] - row['LScore']])
        y_val_data.append(1)
        
        X_val_data.append([row['LTeamID'], row['WTeamID'], row['LScore'] - row['WScore']])
        y_val_data.append(0)
    
    # Convert to numpy arrays
    X_train = np.array(X_train_data, dtype=np.float32)
    y_train = np.array(y_train_data, dtype=np.float32)
    X_val = np.array(X_val_data, dtype=np.float32)
    y_val = np.array(y_val_data, dtype=np.float32)
    
    # Get all unique team IDs
    team_ids = np.unique(np.concatenate([df['WTeamID'].unique(), df['LTeamID'].unique()]))
    
    return X_train, X_val, y_train, y_val, team_ids