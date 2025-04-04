from attempts.section4.data_preprocessing import load_and_preprocess_data
from attempts.section4.evaluate import generate_nerual_net_sub, generate_log_regs_sub, generate_binnary_class_sub
from attempts.section4.train import train_nn_model, train_log_reg_model, train_binary_classification_model


if __name__ == "__main__":
    
    # File paths
    training_data_path = './datasets/MNCAATourneyDetailedResults.csv'
    submission_template_path = './datasets/SampleSubmissionStage1.csv'
    neural_net_model = './neural_net_model.pth'
    log_reg_model = './log_reg_model.pth'
    binary_class_model = './binary_class_model.pth'
        
    X_train, X_test, y_train, y_test, team_ids = load_and_preprocess_data(training_data_path)
    
    # Train the model
    print("Training the model...")
    train_nn_model(training_data_path, num_epochs=20, batch_size=64, learning_rate=0.01)
    train_log_reg_model(training_data_path, num_epochs=20, batch_size=64, learning_rate=0.01)
    train_binary_classification_model(training_data_path, num_epochs=20, batch_size=64, learning_rate=0.01)
    
    # Generate the submission file
    print("Generating the submission file...")
    generate_nerual_net_sub(team_ids, neural_net_model)
    generate_log_regs_sub(team_ids, log_reg_model)
    generate_binnary_class_sub(team_ids, binary_class_model)
    
    print("Submission file generated successfully.")
    