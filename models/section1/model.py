import torch

class StarterModel(torch.nn.Module):
    # integrate ingestor.py into the model
    def __init__(self, ingestor) -> None:
        super(StarterModel, self).__init__()
        self.ingestor = ingestor
        self.seed_df = self.ingestor.get_seed_df_build()
        self.submission_df = self.ingestor.get_submission_df_build()
        
        # Reformat and merge data
        self.reformatted_data = self.ingestor.reformat_and_merge_data_build()  
            
        # Define model layers 
        self.layer1 = torch.nn.Linear(1, 50)
        self.layer2 = torch.nn.Linear(50, 1)
        
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x
    
    
    def calculate_predictions(self) -> None:
        # Calculate seed diff
        self.reformatted_data['SeedDiff'] = self.reformatted_data['SeedValue1'] - self.reformatted_data['SeedValue2']
        
        # Debug prints
        print("SeedValue1:", self.reformatted_data['SeedValue1'].head())
        print("SeedValue2:", self.reformatted_data['SeedValue2'].head())
        print("SeedDiff:", self.reformatted_data['SeedDiff'].head())
        
        # Convert SeedDiff to tensor
        seed_diff_tensor = torch.tensor(self.reformatted_data['SeedDiff'].values, dtype=torch.float32).view(-1, 1)
        
        # Update 'Pred' Column
        self.reformatted_data['Pred'] = self.forward(seed_diff_tensor).detach().numpy()
        
        # Drop unnecessary columns
        self.reformatted_data = self.reformatted_data.drop(columns=['SeedValue1', 'SeedValue2', 'SeedDiff'])

# Example usage
if __name__ == "__main__":
    from ingestor import Ingestor
    ingestor = Ingestor()
    model = StarterModel(ingestor)
    model.calculate_predictions()
    print(model.reformatted_data.head())  # Print the reformatted data to verify