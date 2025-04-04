import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, seed_value=42) -> None:
        self.seed_value = seed_value
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        self.datasets_dir = os.path.join(self.base_dir, 'datasets')
        
        self.set_seeds()
        self.load_data()
        self.process_data()

    def set_seeds(self) -> None:
        os.environ['PYTHONHASHSEED'] = str(self.seed_value)
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        torch.manual_seed(self.seed_value)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed_value)

    def load_data(self) -> None:
        # Load seed data
        w_seeds:pd.DataFrame = pd.read_csv(os.path.join(self.datasets_dir, 'WNCAATourneySeeds.csv'))
        m_seeds:pd.DataFrame = pd.read_csv(os.path.join(self.datasets_dir, 'MNCAATourneySeeds.csv'))
        
        self.seed_df = pd.concat([w_seeds, m_seeds], axis=0).fillna(0.5)
        self.seed_df = self.seed_df[self.seed_df['Season'] >= 2020]

        # Load historical game results
        w_results = pd.read_csv(os.path.join(self.datasets_dir, 'WNCAATourneyCompactResults.csv'))
        m_results = pd.read_csv(os.path.join(self.datasets_dir, 'MNCAATourneyCompactResults.csv'))
        
        self.historical_df = pd.concat([w_results, m_results], axis=0)
        self.historical_df = self.historical_df[self.historical_df['Season'] >= 2020]
        print("Data loaded successfully")

    def extract_seed_value(self, seed_str):
        try:
            return int(seed_str[1:])  # remove letter prefix
        except (ValueError, TypeError):
            return 16

    def process_data(self):
        self.seed_df['SeedValue'] = self.seed_df['Seed'].apply(lambda x: self.extract_seed_value(x))

        # Use unique team Ids from seed_df
        team_ids:np.ndarray = self.seed_df['TeamID'].unique()
        team_ids:np.ndarray = np.sort(team_ids)
        
        self.num_nodes = len(team_ids)
        self.team_id_to_idx = {team_id: idx for idx, team_id in enumerate(team_ids)}

        # Create node features
        node_features:list = []
        for team_id in team_ids:
            seed_values = self.seed_df.loc[self.seed_df['TeamID'] == team_id, 'SeedValue'].values
            
            if len(seed_values) > 0:
                feature = np.mean(seed_values)
            else:
                feature = 16.0
            node_features.append([feature])
        self.node_features = torch.tensor(node_features, dtype=torch.float)

        # Build edge_index from historical games
        edge_list:list = []
        for idx, row in self.historical_df.iterrows():
            winning_team_id = row['WTeamID']
            losing_team_id = row['LTeamID']
            
            if (winning_team_id in self.team_id_to_idx) and (losing_team_id in self.team_id_to_idx):
                win:int = self.team_id_to_idx[winning_team_id]
                loss:int = self.team_id_to_idx[losing_team_id]
                
                edge_list.append([win, loss])
                edge_list.append([loss, win])
                
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Prepare training pairs for historical games
        pair_indices:list = []
        labels:list = []
        
        for idx, row in self.historical_df.iterrows():
            team_w = row['WTeamID']
            team_l = row['LTeamID']
            
            if (team_w in self.team_id_to_idx) and (team_l in self.team_id_to_idx):
                
                w_idx:int = self.team_id_to_idx[team_w]
                l_idx:int = self.team_id_to_idx[team_l]
                
                pair_indices.append([w_idx, l_idx])
                labels.append(1)
                pair_indices.append([l_idx, w_idx])
                labels.append(0)
                
        self.pair_indices = torch.tensor(pair_indices, dtype=torch.long).t()  # [num_samples, 2]
        self.labels = torch.tensor(labels, dtype=torch.float)  # [num_samples]

        # Split into training and validation pairs
        train_idx, val_idx = train_test_split(np.arange(len(self.labels)), test_size=0.2, random_state=self.seed_value, stratify=self.labels.numpy())
        self.train_pairs = self.pair_indices[:, train_idx]
        self.train_labels = self.labels[train_idx]
        self.val_pairs = self.pair_indices[:, val_idx]
        self.val_labels = self.labels[val_idx]