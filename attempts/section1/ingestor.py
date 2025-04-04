import pandas as pd
import numpy as np
import os

class Ingestor:
    def __init__(self) -> None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        datasets_dir = os.path.join(base_dir, 'datasets')
        
        self.w_seed = pd.read_csv(os.path.join(datasets_dir, 'WNCAATourneySeeds.csv'))
        self.m_seed = pd.read_csv(os.path.join(datasets_dir, 'MNCAATourneySeeds.csv'))
        self.seed_df = pd.concat([self.m_seed, self.w_seed], axis=0).fillna(0.05)
        self.submission_df = pd.read_csv(os.path.join(datasets_dir, 'SampleSubmissionStage1.csv'))
        
    def get_seed_df_build(self) -> pd.DataFrame:
        return self.seed_df
    
    def get_submission_df_build(self) -> pd.DataFrame:
        return self.submission_df
    
    @staticmethod
    def extract_game_info(id_str: str) -> tuple[int, int, int]: 
        # Extract year and team ids 
        parts: list[str] = id_str.split('_')
        
        year: int = int(parts[0])
        teamId1: int = int(parts[1])
        teamId2: int = int(parts[2])
        
        return year, teamId1, teamId2 
        
    @staticmethod
    def extract_seed_value(seed_str: str) -> int:
        # Extract seed value
        unselected_seed: int = 16
        try: 
            return int(seed_str[1:])
        # set seed to 16 for unselected
        except ValueError:
            return unselected_seed
    
    def reformat_and_merge_data_build(self) -> pd.DataFrame:    
        # Reformat the data
        self.submission_df[['Season', 'TeamID1', 'TeamID2']] = self.submission_df['ID'].apply(self.extract_game_info).tolist()
        self.seed_df['SeedValue'] = self.seed_df['Seed'].apply(self.extract_seed_value)
        
        # Merge seed information for TeamID1
        self.submission_df = pd.merge(self.submission_df, self.seed_df[['Season', 'TeamID', 'SeedValue']],
                                      left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'],
                                      how='left')
        self.submission_df = self.submission_df.rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

        
        # Merge seed information for TeamID2
        self.submission_df = pd.merge(self.submission_df, self.seed_df[['Season', 'TeamID', 'SeedValue']],
                                      left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'],
                                      how='left')
        self.submission_df = self.submission_df.rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])
        
         # Fill in missing seed values
        self.submission_df = self.submission_df.copy() 
        self.submission_df['SeedValue1'] = self.submission_df['SeedValue1'].fillna(16)
        self.submission_df['SeedValue2'] = self.submission_df['SeedValue2'].fillna(16)
        
        return self.submission_df