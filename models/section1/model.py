import torch
import pandas as pd

class StarterModel(torch.nn.Module):
    # integrate ingestor.py into the model
    def __init__(self, ingestor) -> None:
        super(StarterModel, self).__init__()
        self.ingestor = ingestor
        self.seed_df = self.ingestor.get_seed_df_build()
        self.submission_df = self.ingestor.get_submission_df_build()
        
        self.sigmoid = torch.nn.Sigmoid()
        # Reformat and merge data
        self.reformatted_data = self.ingestor.reformat_and_merge_data_build()  
            
        # Define model layers 
        self.layer1 = torch.nn.Linear(1, 50)
        self.layer2 = torch.nn.Linear(50, 1)
        
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        
        return x
    
    def calculate_predictions(self) -> None:
        # Calculate seed diff
        self.reformatted_data['SeedDiff'] = self.reformatted_data['SeedValue1'] - self.reformatted_data['SeedValue2']
    
        # Convert SeedDiff to tensor
        seed_diff_tensor = torch.tensor(self.reformatted_data['SeedDiff'].values, dtype=torch.float32).view(-1, 1)
        
        # Update 'Pred' Column
        self.reformatted_data['Pred'] = self.forward(seed_diff_tensor).detach().cpu().numpy().flatten()
        
        # Drop unnecessary columns
        self.reformatted_data = self.reformatted_data.drop(columns=['SeedValue1', 'SeedValue2', 'SeedDiff'])


    def generate_all_positive_submissions (self) -> pd.DataFrame:
        # Generate all positive submissions
        self.submission_df['Pred'] = 1.0
        return self.reformatted_data[['ID', 'Pred']]
        
    def generate_all_negative_submissions (self) -> pd.DataFrame:
        # Generate all negative submissions
        self.submission_df['Pred'] = 0.0
        return self.reformatted_data[['ID', 'Pred']]
    
    def generate_most_frequency_submissions (self) -> pd.DataFrame:
        # Generate most frequency submissions
        self.submission_df['Pred'] = self.reformatted_data['Pred'].mean()
        return self.reformatted_data[['ID', 'Pred']]