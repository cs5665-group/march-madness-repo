from competition_code.plotting import plot_training_history

class LossTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        
    def update(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
    
    def plot(self, model_name, save_pth):
        plot_training_history(self.train_losses, self.val_losses, model_name, save_pth)