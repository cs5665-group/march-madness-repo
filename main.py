from models.section1.ingestor import Ingestor
from models.section1.model import StarterModel
from pandas import DataFrame

if __name__ == "__main__":
    ingestor:Ingestor = Ingestor()
    model:StarterModel = StarterModel(ingestor)
    model.calculate_predictions()
    print(model.reformatted_data.head())  # Print the reformatted data to verify


    #Generate all positvie submissions
    positive_submissions:DataFrame = model.generate_all_positive_submissions()
    positive_submissions.to_csv('positive_submissions.csv', index=False)
    
    #Generate all negative submissions
    negative_submissions:DataFrame = model.generate_all_negative_submissions()
    negative_submissions.to_csv('negative_submissions.csv', index=False)
    
    #Generate most frequency submissions
    most_frequency_submissions:DataFrame = model.generate_most_frequency_submissions()
    most_frequency_submissions.to_csv('most_frequency_submissions.csv', index=False)
    
    print("All submissions generated successfully!")