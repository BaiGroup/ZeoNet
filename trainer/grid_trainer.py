import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset.grid_dataset import ZeoGridsDataset
from trainer.base_trainer import BaseTrainer

class GridTrainer(BaseTrainer):
    """Specialized trainer for 3D grid representations"""

    def __init__(self, config):
        super().__init__(config)

    def _setup_training_components(self):
        """Setup training components"""
        # Create dataloaders
        print(f"Loading training data from {self.config.loader.data_path}")
        train_set = ZeoGridsDataset(root_dir=self.config.loader.data_path, 
                                    grid_size=self.config.loader.grid_size, grid_resolution=self.config.loader.grid_resolution, base_grid_resolution=self.config.loader.base_grid_resolution,
                                    is_train=True, is_val=False)
        self.train_loader = DataLoader(train_set, batch_size=self.config.loader.batch_size, shuffle=True)

        val_set = ZeoGridsDataset(root_dir=self.config.loader.data_path, 
                                    grid_size=self.config.loader.grid_size, grid_resolution=self.config.loader.grid_resolution, base_grid_resolution=self.config.loader.base_grid_resolution,
                                    is_train=False, is_val=True)
        self.val_loader = DataLoader(val_set, batch_size=self.config.loader.batch_size, shuffle=False)

        test_set = ZeoGridsDataset(root_dir=self.config.loader.data_path, 
                                       grid_size=self.config.loader.grid_size, grid_resolution=self.config.loader.grid_resolution, base_grid_resolution=self.config.loader.base_grid_resolution,
                                       is_train=False, is_val=False)
        self.test_loader = DataLoader(test_set, batch_size=self.config.loader.batch_size, shuffle=False)
        print(f"training samples: {len(self.train_loader.dataset)}, validation samples: {len(self.val_loader.dataset)}, test samples: {len(self.test_loader.dataset)}")
        
    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_data = 0
        for batch_idx, sample in enumerate(self.train_loader):
            data, target = sample['image'], sample['label']
            data, target = data.to(self.device), target.to(self.device)
            num_data += len(data)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_criterion(output, target)
            total_loss += loss.data
            loss.backward()
            self.optimizer.step()
            eval_score = self.eval_criterion(target.cpu().detach().numpy(), output.cpu().detach().numpy())
        return total_loss/(batch_idx+1.0), eval_score

    def _validate(self):
        """Validate the model"""
        self.model.eval()
        num_data = 0
        zeolites = np.array([], dtype=object).reshape(0)
        scores = np.array([], dtype=np.float32).reshape(0,1) 
        targets = np.array([], dtype=np.int64).reshape(0,1)
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.val_loader):
                data, target= sample['image'], sample['label']
                zeolite = sample['metadata']['zeolite']
                data, target = data.to(self.device), target.to(self.device)
                num_data += len(data)
                output = self.model(data)
                scores = np.concatenate((scores, output.data.cpu().numpy()))
                targets = np.concatenate((targets, target.data.cpu().numpy()))
                zeolites = np.concatenate((zeolites, zeolite))
        total_loss = self.loss_criterion(torch.Tensor(scores), torch.Tensor(targets))
        eval_score = self.eval_criterion(targets, scores)
        # Store predictions for analysis
        predictions = list(zip(zeolites, np.squeeze(targets), np.squeeze(scores)))
        return total_loss, eval_score, predictions
    