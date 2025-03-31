import numpy as np
import pandas as pd
import glob
import os

from typing import Dict, List, Optional, Union

class DatasetBuilder: 
    """Builder class for dataset preparation and transformation."""

    def __init__(self, base_data: Dict[str, pd.DataFrame]): 
        self.data = base_data
        self.teams = None
        self.games = None 
        self.seeds = None
        self.cities = None
        self.submission = None
        
    def build_teams(self) -> 'DatasetBuilder': 
        # build teams dataset by combining teams 
        self.teams = pd.concat([self.data.get('MTeams', pd.DataFrame()), 
                                self.data.get('WTeams', pd.DataFrame())])
        
        teams_spelling = pd.concat([self.data.get('MTeamSpellings', pd.DataFrame()), 
                                    self.data.get('WTeamSpellings', pd.DataFrame())])
        
        if not teams_spelling.empty:
            teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
            teams_spelling.columns = ['TeamID', 'TeamNameCount']
            self.teams = pd.merge(self.teams, teams_spelling, how='left', on=['TeamID'])
        
        return self
    
    def build_seeds_dict(self) -> 'DatasetBuilder':
        """Build a dictionary of tournament seeds."""
        seeds_df = pd.concat([self.data.get('MNCAATourneySeeds', pd.DataFrame()), 
                             self.data.get('WNCAATourneySeeds', pd.DataFrame())])
        
        if not seeds_df.empty:
            self.seeds = {'_'.join(map(str, [int(k1), k2])): int(v[1:3]) 
                         for k1, v, k2 in seeds_df[['Season', 'Seed', 'TeamID']].values}
        else:
            self.seeds = {}
            
        return self
    
    def build_games(self) -> 'DatasetBuilder':
        """Build and transform games dataset from detailed and compact results."""
        # Combine season results
        season_dresults = pd.concat([
            self.data.get('MRegularSeasonDetailedResults', pd.DataFrame()), 
            self.data.get('WRegularSeasonDetailedResults', pd.DataFrame())
        ])
        season_cresults = pd.concat([
            self.data.get('MRegularSeasonCompactResults', pd.DataFrame()), 
            self.data.get('WRegularSeasonCompactResults', pd.DataFrame())
        ])
        
        # Combine tournament results
        tourney_dresults = pd.concat([
            self.data.get('MNCAATourneyDetailedResults', pd.DataFrame()), 
            self.data.get('WNCAATourneyDetailedResults', pd.DataFrame())
        ])
        tourney_cresults = pd.concat([
            self.data.get('MNCAATourneyCompactResults', pd.DataFrame()), 
            self.data.get('WNCAATourneyCompactResults', pd.DataFrame())
        ])
        
        # Add season type labels
        if not season_dresults.empty:
            season_dresults['ST'] = 'S'
        if not season_cresults.empty:
            season_cresults['ST'] = 'S'
        if not tourney_dresults.empty:
            tourney_dresults['ST'] = 'T'
        if not tourney_cresults.empty:
            tourney_cresults['ST'] = 'T'
        
        # Use detailed results if available, otherwise use compact
        games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
        
        if not games.empty:
            # Map location codes
            games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})
            
            # Create IDs for games and teams
            games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
            games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
            games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[0], axis=1)
            games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[1], axis=1)
            games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
            games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
            
            # Add seed information if available
            if self.seeds:
                games['Team1Seed'] = games['IDTeam1'].map(self.seeds).fillna(0)
                games['Team2Seed'] = games['IDTeam2'].map(self.seeds).fillna(0)
                games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed']
            
            # Calculate score differences
            games['ScoreDiff'] = games['WScore'] - games['LScore']
            games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'],r['LTeamID']])[0]==r['WTeamID'] else 0., axis=1)
            games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0. else r['ScoreDiff'], axis=1)
            
            # Fill NaN values
            games = games.fillna(-1)
            
            # Filter for tournament games if needed
            self.games = games[games['ST'] == 'T']
        else:
            self.games = pd.DataFrame()
            
        return self
    
    # Add aggregated statics to games dataset
    def add_score_aggregates(self) -> 'DatasetBuilder': 
        
        if self.games is None or self.games.empty: return self
        
        c_score_col: list[str] = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl',
                        'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl',
                        'LBlk', 'LPF']
        
         # Check if these columns exist in the dataset
        available_cols = [col for col in c_score_col if col in self.games.columns]
        
        if not available_cols:
            return self
            
        c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']
        
        # Group by teams and calculate aggregates
        gb = self.games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in available_cols}).reset_index()
        gb.columns = [''.join(c) + '_c_score' for c in gb.columns]
        
        # Merge aggregates back to games
        self.games = pd.merge(self.games, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')
        
        return self
    
    def prepare_submission(self) -> 'DatasetBuilder':
        """Prepare submission template with features."""
        self.submission = self.data.get('SampleSubmissionStage1', None)
        
        if self.submission is not None:
            self.submission['WLoc'] = 3
            self.submission['Season'] = self.submission['ID'].map(lambda x: x.split('_')[0])
            self.submission['Season'] = self.submission['Season'].astype(int)
            self.submission['Team1'] = self.submission['ID'].map(lambda x: x.split('_')[1])
            self.submission['Team2'] = self.submission['ID'].map(lambda x: x.split('_')[2])
            self.submission['IDTeams'] = self.submission.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
            self.submission['IDTeam1'] = self.submission.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
            self.submission['IDTeam2'] = self.submission.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
            
            # Add seed information if available
            if self.seeds:
                self.submission['Team1Seed'] = self.submission['IDTeam1'].map(self.seeds).fillna(0)
                self.submission['Team2Seed'] = self.submission['IDTeam2'].map(self.seeds).fillna(0)
                self.submission['SeedDiff'] = self.submission['Team1Seed'] - self.submission['Team2Seed']
            
            self.submission = self.submission.fillna(-1)
        
        return self
    
    def add_aggregates_to_submission(self) -> 'DatasetBuilder':
        """Add team performance aggregates to submission data."""
        if self.submission is None or self.games is None:
            return self
            
        # Get column names from games dataset that end with '_c_score'
        agg_cols = [col for col in self.games.columns if col.endswith('_c_score')]
        
        if not agg_cols:
            return self
            
        # Extract the aggregated data
        gb = self.games[['IDTeams'] + agg_cols].drop_duplicates(subset=['IDTeams'])
        
        # Merge with submission
        self.submission = pd.merge(self.submission, gb, how='left', 
                                 left_on='IDTeams', right_on='IDTeams')
        
        return self