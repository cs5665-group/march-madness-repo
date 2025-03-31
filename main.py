from models.section1.ingestor import Ingestor
from models.section1.model import StarterModel

<<<<<<< Updated upstream
=======
if __name__ == "__main__":
    
    # File paths
    training_data_path = './datasets/MNCAATourneyDetailedResults.csv'
    submission_template_path = './datasets/SampleSubmissionStage1.csv'
    model_path = './matchup_prediction_model.pth'
        
    X_train, X_test, y_train, y_test, team_ids = load_and_preprocess_data(training_data_path)
    
    # Train the model
    print("Training the model...")
    train_model(training_data_path, num_epochs=20, batch_size=64, learning_rate=0.001)
    
    # Generate the submission file
    print("Generating the submission file...")
    generate_submission(team_ids, model_path)
    print("Submission file generated successfully.")
    
>>>>>>> Stashed changes

if __name__ == "__main__":
    ingestor:Ingestor = Ingestor()
    model:StarterModel = StarterModel(ingestor)
    model.calculate_predictions()
    print(model.reformatted_data.head())  # Print the reformatted data to verify
