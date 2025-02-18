import os 
import pandas as pd

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)), "datasets"))

# Utility function to load dataset
def load_dataset(filename:str) -> pd.DataFrame:
    file_path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    return pd.read_csv(file_path)