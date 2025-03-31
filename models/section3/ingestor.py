import pandas as pd
import glob
import os
from typing import Dict, List, Optional, Union
from models.section3.datasetbuilder import DatasetBuilder

class Ingestor:
    """Main class for loading and preparing tournament data."""
    
    def __init__(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        self.datasets_path = os.path.join(self.base_dir, 'datasets')
        self.data = {}
        self.builder = None
        
    def load_data(self, path: Optional[str] = None) -> 'Ingestor':
        """Load all CSV files from the specified path."""
        if path is None:
            path = os.path.join(self.datasets_path, '**/*.csv')
        
        self.data = {os.path.basename(p).split('.')[0]: pd.read_csv(p, encoding='latin-1') 
                    for p in glob.glob(path, recursive=True)}
        
        return self
    
    def get_builder(self) -> DatasetBuilder:
        """Create and return a DatasetBuilder with loaded data."""
        if not self.data:
            self.load_data()
        
        self.builder = DatasetBuilder(self.data)
        return self.builder
    
    def build_complete_dataset(self) -> Dict[str, Union[pd.DataFrame, dict]]:
        """Build and return a complete processed dataset."""
        builder = self.get_builder()
        
        builder.build_teams() \
               .build_seeds_dict() \
               .build_games() \
               .add_score_aggregates() \
               .prepare_submission() \
               .add_aggregates_to_submission()
        
        return {
            'teams': builder.teams,
            'games': builder.games,
            'seeds': builder.seeds,
            'submission': builder.submission
        }
    
    def get_feature_columns(self) -> List[str]:
        """Get the list of feature columns that could be used for modeling."""
        if self.builder is None or self.builder.games is None:
            return []
            
        exclude_cols = ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2', 
                        'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 
                        'ScoreDiffNorm', 'WLoc']
        
        # Get score columns if they exist
        score_cols = []
        if self.builder.games is not None and not self.builder.games.empty:
            for col in self.builder.games.columns:
                if col.startswith(('W', 'L')) and col[1:] in ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 
                                                             'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 
                                                             'Blk', 'PF']:
                    score_cols.append(col)
        
        exclude_cols.extend(score_cols)
        
        return [c for c in self.builder.games.columns if c not in exclude_cols]