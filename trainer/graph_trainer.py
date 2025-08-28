import torch
from torch.utils.data import DataLoader
import numpy as np
from functools import partial

from dataset.graph_dataset import ZeoCGCNNGraphDataset, ZeoMEGNETGraphDataset, ZeoMACEGraphDataset
from trainer.base_trainer import BaseTrainer
from utils.data_utils import collate_pool, collate_fn, compute_avg_num_neighbors
from dgl.dataloading import GraphDataLoader
from mace.tools import torch_geometric

class CGCNNTrainer(BaseTrainer):
    """Specialized trainer for CGCNN representations"""

    def __init__(self, config):
        super().__init__(config)

    def _setup_training_components(self):
        """Setup training components"""
        # Create dataloaders
        print(f"Loading training data from {self.config.loader.data_path}")
        train_set = ZeoCGCNNGraphDataset(root_dir=self.config.loader.data_path, 
                                    max_num_nbr=self.config.loader.max_num_nbr, radius=self.config.loader.radius, step=self.config.loader.step,
                                    is_train=True, is_val=False)
        self.train_loader = DataLoader(train_set, collate_fn=collate_pool,batch_size=self.config.loader.batch_size, shuffle=True)

        val_set = ZeoCGCNNGraphDataset(root_dir=self.config.loader.data_path, 
                                    max_num_nbr=self.config.loader.max_num_nbr, radius=self.config.loader.radius, step=self.config.loader.step,
                                    is_train=False, is_val=True)
        self.val_loader = DataLoader(val_set, collate_fn=collate_pool, batch_size=self.config.loader.batch_size, shuffle=False)

        test_set = ZeoCGCNNGraphDataset(root_dir=self.config.loader.data_path, 
                                       max_num_nbr=self.config.loader.max_num_nbr, radius=self.config.loader.radius, step=self.config.loader.step,
                                       is_train=False, is_val=False)
        self.test_loader = DataLoader(test_set, collate_fn=collate_pool, batch_size=self.config.loader.batch_size, shuffle=False)

        print(f"training samples: {len(self.train_loader.dataset)}, validation samples: {len(self.val_loader.dataset)}, test samples: {len(self.test_loader.dataset)}")
    
    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_data = 0
        for i, (input, target, _) in enumerate(self.train_loader):
            if self.device.type == 'cuda':
                input_var = (input[0].cuda(non_blocking=True),
                            input[1].cuda(non_blocking=True),
                            input[2].cuda(non_blocking=True),
                            [crystal_idx.cuda(non_blocking=True) for crystal_idx in input[3]])
                target_var = target.cuda(non_blocking=True)
            else:
                input_var = (input[0], input[1], input[2], input[3])
                target_var = target
            self.optimizer.zero_grad()
            output = self.model(*input_var)
            loss = self.loss_criterion(output, target_var)
            total_loss += loss.data
            loss.backward()
            self.optimizer.step()
            eval_score = self.eval_criterion(target_var.cpu().detach().numpy(), output.cpu().detach().numpy())
        return total_loss/(i+1.0), eval_score

    def _validate(self):
        """Validate the model"""
        self.model.eval()
        zeolites = np.array([], dtype=object).reshape(0)
        scores = np.array([], dtype=np.float32).reshape(0,1) 
        targets = np.array([], dtype=np.int64).reshape(0,1)
        with torch.no_grad():
            for i, (input, target, batch_cif_ids) in enumerate(self.val_loader):
                zeolite = batch_cif_ids
                if self.device.type == 'cuda':
                    input_var = (input[0].cuda(non_blocking=True),
                                input[1].cuda(non_blocking=True),
                                input[2].cuda(non_blocking=True),
                                [crystal_idx.cuda() for crystal_idx in input[3]])
                    target_var = target.cuda(non_blocking=True)
                else:
                    input_var = (input[0], input[1], input[2], input[3])
                    target_var = target
                output = self.model(*input_var)
                scores = np.concatenate((scores, output.data.cpu().numpy()))
                targets = np.concatenate((targets, target_var.data.cpu().numpy()))
                zeolites = np.concatenate((zeolites, zeolite))
        total_loss = self.loss_criterion(torch.Tensor(scores), torch.Tensor(targets))
        eval_score = self.eval_criterion(targets, scores)
        # Store predictions for analysis
        predictions = list(zip(zeolites, np.squeeze(targets), np.squeeze(scores)))
        return total_loss, eval_score, predictions
    
class MEGNETTrainer(BaseTrainer):
    """Specialized trainer for CGCNN representations"""

    def __init__(self, config):
        super().__init__(config)

    def _setup_training_components(self):
        """Setup training components"""
        # Create dataloaders
        print(f"Loading training data from {self.config.loader.data_path}")
        collate_fn_graph = partial(collate_fn, include_line_graph=False)
        train_set = ZeoMEGNETGraphDataset(root_dir=self.config.loader.data_path, cutoff=self.config.loader.cutoff,
                                         is_train=True, is_val=False)
        self.train_loader = GraphDataLoader(train_set, collate_fn=collate_fn_graph, batch_size=self.config.loader.batch_size, shuffle=True)

        val_set = ZeoMEGNETGraphDataset(root_dir=self.config.loader.data_path, cutoff=self.config.loader.cutoff,
                                       is_train=False, is_val=True)
        self.val_loader = GraphDataLoader(val_set, collate_fn=collate_fn_graph, batch_size=self.config.loader.batch_size, shuffle=False)

        test_set = ZeoMEGNETGraphDataset(root_dir=self.config.loader.data_path, cutoff=self.config.loader.cutoff,
                                         is_train=False, is_val=False)
        self.test_loader = GraphDataLoader(test_set, collate_fn=collate_fn_graph, batch_size=self.config.loader.batch_size, shuffle=False)

        print(f"training samples: {len(self.train_loader.dataset)}, validation samples: {len(self.val_loader.dataset)}, test samples: {len(self.test_loader.dataset)}")

    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_data = 0

        for i, (g, lat, labels, state_attrs, batch_cif_ids) in enumerate(self.train_loader):
            g.edata["lattice"] = torch.repeat_interleave(lat, g.batch_num_edges(), dim=0)
            g.edata["pbc_offshift"] = (g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
            g.ndata["pos"] = (
                g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lat, g.batch_num_nodes(), dim=0)
            ).sum(dim=1)

            g = g.to(self.device)
            labels = labels.to(self.device, non_blocking=(self.device.type == 'cuda'))
            state_attrs = state_attrs.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(g, state_attrs)
            loss = self.loss_criterion(output, labels)
            total_loss += loss.data
            loss.backward()
            self.optimizer.step()
            eval_score = self.eval_criterion(labels.cpu().detach().numpy(), output.cpu().detach().numpy())
        return total_loss/(i+1.0), eval_score

    def _validate(self):
        """Validate the model"""
        self.model.eval()
        zeolites = np.array([], dtype=object).reshape(0)
        scores = np.array([], dtype=np.float32).reshape(0,1) 
        targets = np.array([], dtype=np.int64).reshape(0,1)
        with torch.no_grad():
            for i, (g, lat, labels, state_attrs, batch_cif_ids) in enumerate(self.val_loader):
                g.edata["lattice"] = torch.repeat_interleave(lat, g.batch_num_edges(), dim=0)
                g.edata["pbc_offshift"] = (g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
                g.ndata["pos"] = (
                    g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lat, g.batch_num_nodes(), dim=0)
                ).sum(dim=1)

                g = g.to(self.device)
                labels = labels.to(self.device, non_blocking=(self.device.type == 'cuda'))
                state_attrs = state_attrs.to(self.device)

                output = self.model(g, state_attrs)
                scores = np.concatenate((scores, output.data.cpu().numpy()))
                targets = np.concatenate((targets, labels.data.cpu().numpy()))
                zeolites = np.concatenate((zeolites, batch_cif_ids))

        total_loss = self.loss_criterion(torch.Tensor(scores), torch.Tensor(targets))
        eval_score = self.eval_criterion(targets, scores)
        predictions = list(zip(zeolites, np.squeeze(targets), np.squeeze(scores)))
        return total_loss, eval_score, predictions

class M3GNETTrainer(BaseTrainer):
    """Specialized trainer for M3GNET representations"""

    def __init__(self, config):
        super().__init__(config)

    def _setup_training_components(self):
        """Setup training components"""
        # Create dataloaders
        print(f"Loading training data from {self.config.loader.data_path}")
        collate_fn_graph = partial(collate_fn, include_line_graph=True)
        train_set = ZeoMEGNETGraphDataset(root_dir=self.config.loader.data_path, 
                                         cutoff=self.config.loader.cutoff, threebody_cutoff=self.config.loader.threebody_cutoff,
                                         include_line_graph=True, directed_line_graph=False,
                                         is_train=True, is_val=False)
        self.train_loader = GraphDataLoader(train_set, collate_fn=collate_fn_graph, batch_size=self.config.loader.batch_size, shuffle=True)

        val_set = ZeoMEGNETGraphDataset(root_dir=self.config.loader.data_path, 
                                       cutoff=self.config.loader.cutoff, threebody_cutoff=self.config.loader.threebody_cutoff,
                                       include_line_graph=True, directed_line_graph=False,
                                       is_train=False, is_val=True)
        self.val_loader = GraphDataLoader(val_set, collate_fn=collate_fn_graph, batch_size=self.config.loader.batch_size, shuffle=False)

        test_set = ZeoMEGNETGraphDataset(root_dir=self.config.loader.data_path, 
                                        cutoff=self.config.loader.cutoff, threebody_cutoff=self.config.loader.threebody_cutoff,
                                        include_line_graph=True, directed_line_graph=False,
                                        is_train=False, is_val=False)
        self.test_loader = GraphDataLoader(test_set, collate_fn=collate_fn_graph, batch_size=self.config.loader.batch_size, shuffle=False)

        print(f"training samples: {len(self.train_loader.dataset)}, validation samples: {len(self.val_loader.dataset)}, test samples: {len(self.test_loader.dataset)}")

    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_data = 0

        for i, (g, lat, labels, state_attrs, lg, batch_cif_ids) in enumerate(self.train_loader):

            g.edata["lattice"] = torch.repeat_interleave(lat, g.batch_num_edges(), dim=0)
            g.edata["pbc_offshift"] = (g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
            g.ndata["pos"] = (
                g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lat, g.batch_num_nodes(), dim=0)
            ).sum(dim=1)
            
            g = g.to(self.device)
            labels = labels.to(self.device, non_blocking=(self.device.type == 'cuda'))
            state_attrs = state_attrs.to(self.device)
            lg = lg.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(g, state_attrs, lg)
            loss = self.loss_criterion(output, labels)
            total_loss += loss.data
            loss.backward()
            self.optimizer.step()
            eval_score = self.eval_criterion(labels.cpu().detach().numpy(), output.cpu().detach().numpy())

        return total_loss/(i+1.0), eval_score

    def _validate(self):
        """Validate the model"""
        self.model.eval()
        zeolites = np.array([], dtype=object).reshape(0)
        scores = np.array([], dtype=np.float32).reshape(0,1) 
        targets = np.array([], dtype=np.int64).reshape(0,1)

        with torch.no_grad():
            for i, (g, lat, labels, state_attrs, lg, batch_cif_ids) in enumerate(self.val_loader):
                g.edata["lattice"] = torch.repeat_interleave(lat, g.batch_num_edges(), dim=0)
                g.edata["pbc_offshift"] = (g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
                g.ndata["pos"] = (
                    g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lat, g.batch_num_nodes(), dim=0)
                ).sum(dim=1)

                g = g.to(self.device)
                labels = labels.to(self.device, non_blocking=(self.device.type == 'cuda'))
                state_attrs = state_attrs.to(self.device)
                lg = lg.to(self.device)

                output = self.model(g, state_attrs, lg)
                scores = np.concatenate((scores, output.data.cpu().numpy()))
                targets = np.concatenate((targets, labels.data.cpu().numpy()))
                zeolites = np.concatenate((zeolites, batch_cif_ids))

        total_loss = self.loss_criterion(torch.Tensor(scores), torch.Tensor(targets))
        eval_score = self.eval_criterion(targets, scores)
        predictions = list(zip(zeolites, np.squeeze(targets), np.squeeze(scores)))
        return total_loss, eval_score, predictions
    
class MACETrainer(BaseTrainer):
    """Specialized trainer for MACE representations"""

    def __init__(self, config):
        super().__init__(config)

    def _setup_training_components(self):
        """Setup training components"""
        # Create dataloaders
        print(f"Loading training data from {self.config.loader.data_path}")
        train_set = ZeoMACEGraphDataset(root_dir=self.config.loader.data_path, cutoff=self.config.loader.cutoff,
                                        is_train=True, is_val=False)
        self.train_loader = torch_geometric.dataloader.DataLoader(train_set, batch_size=self.config.loader.batch_size, shuffle=True)

        val_set = ZeoMACEGraphDataset(root_dir=self.config.loader.data_path, cutoff=self.config.loader.cutoff,
                                    is_train=False, is_val=True)
        self.val_loader = torch_geometric.dataloader.DataLoader(val_set, batch_size=self.config.loader.batch_size, shuffle=False)

        test_set = ZeoMACEGraphDataset(root_dir=self.config.loader.data_path, cutoff=self.config.loader.cutoff,
                                       is_train=False, is_val=False)

        self.test_loader = torch_geometric.dataloader.DataLoader(test_set, batch_size=self.config.loader.batch_size, shuffle=False)

        print(f"training samples: {len(self.train_loader.dataset)}, validation samples: {len(self.val_loader.dataset)}, test samples: {len(self.test_loader.dataset)}")
    
    def _train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_data = 0
        for i, sample in enumerate(self.train_loader):
            atomicdata, target = sample['AtomicData'], sample['label']

            atomicdata = atomicdata.to(self.device)
            target = target.to(self.device, non_blocking=(self.device.type == 'cuda'))

            self.optimizer.zero_grad()
            output = self.model(atomicdata, training=True)
            loss = self.loss_criterion(output, target)
            total_loss += loss.data
            loss.backward()
            self.optimizer.step()
            eval_score = self.eval_criterion(target.cpu().detach().numpy(), output.cpu().detach().numpy())
        return total_loss/(i+1.0), eval_score
    
    def _validate(self):
        """Validate the model"""
        self.model.eval()
        zeolites = np.array([], dtype=object).reshape(0)
        scores = np.array([], dtype=np.float32).reshape(0,1) 
        targets = np.array([], dtype=np.int64).reshape(0,1)

        with torch.no_grad():
            for i, sample in enumerate(self.val_loader):
                atomicdata, target, zeolite = sample['AtomicData'], sample['label'], sample['zeolite']

                atomicdata = atomicdata.to(self.device)
                target = target.to(self.device, non_blocking=(self.device.type == 'cuda'))

                output = self.model(atomicdata, training=False)
                scores = np.concatenate((scores, output.data.cpu().numpy()))
                targets = np.concatenate((targets, target.data.cpu().numpy()))
                zeolites = np.concatenate((zeolites, zeolite))
                
        total_loss = self.loss_criterion(torch.Tensor(scores), torch.Tensor(targets))
        eval_score = self.eval_criterion(targets, scores)
        predictions = list(zip(zeolites, np.squeeze(targets), np.squeeze(scores)))
        return total_loss, eval_score, predictions
