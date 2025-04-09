from attempts.section4.data_preprocessing import load_and_preprocess_data
from attempts.section4.evaluate import generate_nerual_net_sub, generate_log_regs_sub, generate_binnary_class_sub
from attempts.section4.train import train_nn_model, train_log_reg_model, train_binary_classification_model
from attempts.section4.plotting import visualize_data_distribution


if __name__ == "__main__":
    
    # File paths
    training_data_path = './datasets/MNCAATourneyDetailedResults.csv'
    submission_template_path = './datasets/SampleSubmissionStage1.csv'
    neural_net_path = './neural_net_model.pth'
    log_reg_path = './log_reg_model.pth'
    binary_class_path = './binary_class_model.pth'
    
    # Create visualizations of the data
    print("Creating data visualizations...")
    visualize_data_distribution(training_data_path)
        
    # Load data
    X_train, X_test, y_train, y_test, team_ids = load_and_preprocess_data(training_data_path)
    
    # Train the models with anti-overfitting configurations
    print("Training the models...")
    nn_model, nn_brier, nn_acc = train_nn_model(
        training_data_path, 
        num_epochs=20, 
        batch_size=64, 
        learning_rate=0.01,
        dropout_rate=0.7
    )
    
    log_model, log_brier, log_acc = train_log_reg_model(
        training_data_path, 
        num_epochs=20, 
        batch_size=64, 
        learning_rate=0.01
    )
    
    bin_model, bin_brier, bin_acc = train_binary_classification_model(
        training_data_path, 
        num_epochs=20, 
        batch_size=64, 
        learning_rate=0.01
    )
    
    # Generate the submission files
    print("Generating tournament prediction files...")
    generate_nerual_net_sub(team_ids, neural_net_path)
    generate_log_regs_sub(team_ids, log_reg_path)
    generate_binnary_class_sub(team_ids, binary_class_path)
    
    # Print summary of model performance
    print("\n=== Model Performance Summary ===")
    print(f"Neural Network:        Brier Score = {nn_brier:.4f}, Accuracy = {nn_acc:.4f}")
    print(f"Logistic Regression:   Brier Score = {log_brier:.4f}, Accuracy = {log_acc:.4f}")
    print(f"Binary Classification: Brier Score = {bin_brier:.4f}, Accuracy = {bin_acc:.4f}")
    
    print("All visualizations saved to the 'visualizations' directory.")
    print("Tournament prediction files generated successfully.")