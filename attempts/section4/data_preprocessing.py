import pandas as pd
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