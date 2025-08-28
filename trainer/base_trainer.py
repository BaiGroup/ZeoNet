from typing import Any, Dict, Optional, Tuple, List, Union
from pathlib import Path
import torch
import torch.nn as nn
import os
import pandas as pd
from sklearn.metrics import r2_score
from torch.optim import Adam
from models.model import get_model

class BaseTrainer():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if config.training.use_gpu and torch.cuda.is_available() else 'cpu')
        # Initialize model
        self.model = get_model(config)
        self.model = self.model.to(self.device)
        self.resume_path = self.config.training.resume_path
        self.save_dir = self.config.training.save_dir
        self.optimizer = Adam(self.model.parameters(), lr=self.config.training.learning_rate,
                         weight_decay=self.config.training.weight_decay)    
        self.loss_criterion = nn.MSELoss()
        self.eval_criterion = r2_score
        # Setup training components
        self._setup_training_components()
    
    def _setup_training_components(self):
        """Setup training components - to be implemented by subclasses"""
        raise NotImplementedError

    def _save_checkpoint(self, model, name):
        """Save model checkpoint"""
        checkpoint = model.state_dict()
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save latest/best checkpoint
        save_path = os.path.join(save_dir, name)
        torch.save(checkpoint, save_path)

    def _save_predictions(self, predictions: List[Tuple[str, float, float]]):
        """Save predictions to a CSV file"""
        df = pd.DataFrame(predictions, columns=['zeolite', 'target', 'prediction'])
        df.to_csv(os.path.join(self.save_dir, 'best_predictions.csv'), index=False)
    
    def _train_epoch(self):
        """Training loop - to be implemented by subclasses"""
        raise NotImplementedError 
    
    def _validate(self):
        """Training loop - to be implemented by subclasses"""
        raise NotImplementedError 
    
    def train(self):
        """Run the training process"""
        # Load checkpoint if resuming
        if self.resume_path and os.path.isfile(self.resume_path):
            print(f"Resuming training from checkpoint: {self.resume_path}")
            checkpoint = torch.load(self.resume_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
        else:
            print("Training from scratch.")

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(1, self.config.training.max_epochs+1):
            self.current_epoch = epoch
            train_loss, train_r2 = self._train_epoch()
            val_loss, val_r2, predictions = self._validate()
            print(f"Epoch {epoch} - Train Loss of Last Batch: {train_loss:.4f} - Train r2 of Last Batch: {train_r2:.4f} - Val Loss: {val_loss:.4f} - Val r2: {val_r2:.4f}")
            self._save_checkpoint(self.model, "latest_model.pth")
            # Save checkpoint and predictions if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(self.model, "best_model.pth")
                self._save_predictions(predictions)
