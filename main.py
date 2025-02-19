from models.section1.ingestor import Ingestor
from models.section1.model import StarterModel


if __name__ == "__main__":
    ingestor:Ingestor = Ingestor()
    model:StarterModel = StarterModel(ingestor)
    model.calculate_predictions()
    print(model.reformatted_data.head())  # Print the reformatted data to verify
