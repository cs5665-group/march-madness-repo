import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from torch.utils.data import Dataset, DataLoader, TensorDataset


def plot_training_history(train_losses, val_losses, model_name='Model', save_path=None):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        model_name: Name of the model for the title
        save_path: Path to save the figure (if None, just display)
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add text showing final loss values
    plt.annotate(f'Final train loss: {train_losses[-1]:.4f}', 
                xy=(0.02, 0.95), xycoords='axes fraction')
    plt.annotate(f'Final val loss: {val_losses[-1]:.4f}', 
                xy=(0.02, 0.90), xycoords='axes fraction')
    
    # Calculate and display gap between training and validation
    gap = val_losses[-1] - train_losses[-1]
    plt.annotate(f'Gap: {gap:.4f}', xy=(0.02, 0.85), xycoords='axes fraction',
                color='red' if gap > 0.1 else 'green')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name='Model', save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for the title
        save_path: Path to save the figure (if None, just display)
    """
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # calculate and display metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    plt.figtext(0.02, 0.1, f"Accuracy: {accuracy:.4f}", fontsize=10)
    plt.figtext(0.02, 0.07, f"Precision: {precision:.4f}", fontsize=10)
    plt.figtext(0.02, 0.04, f"Recall: {recall:.4f}", fontsize=10)
    plt.figtext(0.02, 0.01, f"F1 Score: {f1:.4f}", fontsize=10)
    
    if save_path: 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()
    
    
def plot_roc_curve(y_true, y_pred, model_name='Model', save_path=None):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        model_name: Name of the model for the title
        save_path: Path to save the figure
    """
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_precision_recall_curve(y_true, y_pred, model_name='Model', save_path=None):
    """
    Plot precision-recall curve
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        model_name: Name of the model for the title
        save_path: Path to save the figure
    """
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate average precision
    avg_precision = np.mean(precision)
    plt.annotate(f'Avg Precision: {avg_precision:.4f}', 
                xy=(0.02, 0.95), xycoords='axes fraction')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Precision-Recall curve saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_team_embeddings(model, num_teams, team_names=None, model_name='Model', save_path=None):
    """
    Plot team embeddings in 2D space using PCA
    
    Args:
        model: Trained model with team_embedding layer
        num_teams: Number of teams
        team_names: Dictionary mapping team IDs to names (optional)
        model_name: Name of the model for the title
        save_path: Path to save the figure
    """
    from sklearn.decomposition import PCA
    
    # Get team embeddings from model
    with torch.no_grad():
        team_ids = torch.arange(num_teams, dtype=torch.long)
        embeddings = model.team_embedding(team_ids).cpu().numpy()
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    # Add team names if provided
    if team_names:
        for i, (x, y) in enumerate(embeddings_2d):
            if i in team_names:
                plt.annotate(team_names[i], (x, y), fontsize=8)
    
    plt.title(f'{model_name} Team Embeddings (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    explained_variance = pca.explained_variance_ratio_
    plt.figtext(0.02, 0.02, f"Explained variance: {sum(explained_variance):.4f}", fontsize=10)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Team embeddings plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_prediction_distribution(y_pred, model_name='Model', save_path=None):
    """
    Plot the distribution of predicted probabilities
    
    Args:
        y_pred: Predicted probabilities
        model_name: Name of the model for the title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(y_pred, bins=50, kde=True)
    plt.title(f'{model_name} Prediction Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    
    # Add vertical lines at 0.5
    plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Threshold')
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Prediction distribution plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_calibration_curve(y_true, y_pred, model_name='Model', n_bins=10, save_path=None):
    """
    Plot calibration curve (reliability diagram)
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        model_name: Name of the model for the title
        n_bins: Number of bins for the histogram
        save_path: Path to save the figure
    """
    from sklearn.calibration import calibration_curve
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.plot(prob_pred, prob_true, 's-', label='Model')
    
    plt.title(f'{model_name} Calibration Curve')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate and display Brier score
    from sklearn.metrics import brier_score_loss
    brier = brier_score_loss(y_true, y_pred)
    plt.annotate(f'Brier Score: {brier:.4f}', 
                xy=(0.02, 0.95), xycoords='axes fraction')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Calibration curve saved to {save_path}")
    else:
        plt.show()
    plt.close()

def evaluate_and_visualize_model(model, X_test, y_test, model_name='Model', save_dir='visualizations'):
    """
    Evaluate model and create all visualization plots
    
    Args:
        model: Trained PyTorch model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for plot titles
        save_dir: Directory to save plots
    """
    # Convert data to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create test data loader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Make predictions
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds = model(X_batch).squeeze().cpu().numpy()
            all_preds.append(preds)
    
    # Convert predictions to numpy array
    y_pred = np.concatenate(all_preds)
    
    # Create directory for visualizations
    os.makedirs(save_dir, exist_ok=True)
    
    # Create and save all plots
    plot_confusion_matrix(y_test, y_pred, model_name, 
                        save_path=f"{save_dir}/{model_name}_confusion_matrix.png")
    
    plot_roc_curve(y_test, y_pred, model_name, 
                save_path=f"{save_dir}/{model_name}_roc_curve.png")
    
    plot_precision_recall_curve(y_test, y_pred, model_name, 
                              save_path=f"{save_dir}/{model_name}_pr_curve.png")
    
    plot_prediction_distribution(y_pred, model_name, 
                               save_path=f"{save_dir}/{model_name}_prediction_distribution.png")
    
    plot_calibration_curve(y_test, y_pred, model_name, 
                         save_path=f"{save_dir}/{model_name}_calibration_curve.png")
    
    # Plot team embeddings if the model has them
    try:
        num_teams = model.team_embedding.num_embeddings
        plot_team_embeddings(model, num_teams, model_name=model_name, 
                           save_path=f"{save_dir}/{model_name}_team_embeddings.png")
    except AttributeError:
        print("Model doesn't have team embeddings, skipping embedding visualization")
    
    return {
        'predictions': y_pred,
        'confusion_matrix': f"{save_dir}/{model_name}_confusion_matrix.png",
        'roc_curve': f"{save_dir}/{model_name}_roc_curve.png",
        'pr_curve': f"{save_dir}/{model_name}_pr_curve.png",
        'prediction_distribution': f"{save_dir}/{model_name}_prediction_distribution.png",
        'calibration_curve': f"{save_dir}/{model_name}_calibration_curve.png",
        'team_embeddings': f"{save_dir}/{model_name}_team_embeddings.png"
    }


def visualize_data_distribution(filepath, save_dir='visualizations'):
    """
    Visualize characteristics of the NCAA tournament dataset
    """
    # Load original data
    df = pd.read_csv(filepath)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Score Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(df['WScore'], bins=30, alpha=0.5, label='Winning Team Score')
    plt.hist(df['LScore'], bins=30, alpha=0.5, label='Losing Team Score')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Winning and Losing Scores')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{save_dir}/score_distribution.png")
    plt.close()
    
    # 2. Score Difference Distribution
    score_diff = df['WScore'] - df['LScore']
    plt.figure(figsize=(12, 6))
    plt.hist(score_diff, bins=30)
    plt.xlabel('Score Difference (Winner - Loser)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Score Differences')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axvline(score_diff.mean(), color='red', linestyle='--', 
                label=f'Mean: {score_diff.mean():.2f}')
    plt.legend()
    plt.savefig(f"{save_dir}/score_diff_distribution.png")
    plt.close()
    
    # 3. Team Win Distribution (top 20 teams)
    team_wins = df['WTeamID'].value_counts().reset_index()
    team_wins.columns = ['TeamID', 'Wins']
    team_losses = df['LTeamID'].value_counts().reset_index()
    team_losses.columns = ['TeamID', 'Losses']
    
    # Merge to get total games and win percentage
    team_stats = pd.merge(team_wins, team_losses, on='TeamID', how='outer').fillna(0)
    team_stats['Total'] = team_stats['Wins'] + team_stats['Losses']
    team_stats['WinPct'] = team_stats['Wins'] / team_stats['Total']
    
    # Plot win percentage for top 20 teams by total games
    top_teams = team_stats.sort_values('Total', ascending=False).head(20)
    
    plt.figure(figsize=(14, 7))
    bars = plt.bar(top_teams['TeamID'].astype(str), top_teams['WinPct'])
    plt.xlabel('Team ID')
    plt.ylabel('Win Percentage')
    plt.title('Win Percentage for Top 20 Teams by Total Games')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add win-loss record as labels
    for i, bar in enumerate(bars):
        team = top_teams.iloc[i]
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{int(team['Wins'])}-{int(team['Losses'])}",
            ha='center', va='bottom', rotation=90, fontsize=8
        )
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/team_win_percentage.png")
    plt.close()
    
    # 4. Feature correlations with winning
    # Create binary features
    feature_df = df.copy()
    feature_df['ScoreDiff'] = feature_df['WScore'] - feature_df['LScore']
    feature_df['FGPct_W'] = feature_df['WFGM'] / feature_df['WFGA']
    feature_df['FGPct_L'] = feature_df['LFGM'] / feature_df['LFGA']
    feature_df['ThreePtPct_W'] = feature_df['WFGM3'] / feature_df['WFGA3']
    feature_df['ThreePtPct_L'] = feature_df['LFGM3'] / feature_df['LFGA3']
    feature_df['FTPct_W'] = feature_df['WFTM'] / feature_df['WFTA']
    feature_df['FTPct_L'] = feature_df['LFTM'] / feature_df['LFTA']
    
    # Select relevant columns for correlation
    corr_features = [
        'ScoreDiff', 'FGPct_W', 'FGPct_L', 'ThreePtPct_W', 'ThreePtPct_L',
        'FTPct_W', 'FTPct_L', 'WOR', 'WDR', 'LOR', 'LDR',
        'WAst', 'LAst', 'WTO', 'LTO', 'WStl', 'LStl', 'WBlk', 'LBlk'
    ]
    
    # Calculate correlations with score difference
    corr_data = feature_df[corr_features].corr()['ScoreDiff'].sort_values(ascending=False).drop('ScoreDiff')
    
    # Plot correlations
    plt.figure(figsize=(12, 8))
    sns.barplot(x=corr_data.values, y=corr_data.index)
    plt.title('Feature Correlations with Score Difference')
    plt.xlabel('Correlation Coefficient')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_correlations.png")
    plt.close()
    
    # 5. Season trends
    season_data = feature_df.groupby('Season').agg({
        'ScoreDiff': 'mean',
        'WScore': 'mean',
        'LScore': 'mean',
        'FGPct_W': 'mean',
        'ThreePtPct_W': 'mean'
    }).reset_index()
    
    # Plot score trends over seasons
    plt.figure(figsize=(12, 6))
    plt.plot(season_data['Season'], season_data['WScore'], marker='o', label='Winning Score')
    plt.plot(season_data['Season'], season_data['LScore'], marker='o', label='Losing Score')
    plt.xlabel('Season')
    plt.ylabel('Average Score')
    plt.title('Average Scores by Season')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/season_score_trends.png")
    plt.close()
    
    # Plot shooting percentage trends
    plt.figure(figsize=(12, 6))
    plt.plot(season_data['Season'], season_data['FGPct_W'], marker='o', label='FG%')
    plt.plot(season_data['Season'], season_data['ThreePtPct_W'], marker='o', label='3PT%')
    plt.xlabel('Season')
    plt.ylabel('Percentage')
    plt.title('Shooting Percentages by Season')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/season_shooting_trends.png")
    plt.close()
    
    print(f"Data visualizations saved to {save_dir}/")
    return {
        'score_distribution': f"{save_dir}/score_distribution.png",
        'score_diff_distribution': f"{save_dir}/score_diff_distribution.png",
        'team_win_percentage': f"{save_dir}/team_win_percentage.png",
        'feature_correlations': f"{save_dir}/feature_correlations.png",
        'season_score_trends': f"{save_dir}/season_score_trends.png",
        'season_shooting_trends': f"{save_dir}/season_shooting_trends.png"
    }