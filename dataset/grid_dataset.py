import os
import numpy as np

from utils.data_utils import coordinates_3d

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import h5py
import csv

class ZeoGridsDataset(Dataset):
    ''' Zeolites dataset'''

    def __init__(self, root_dir, grid_size=100, grid_resolution=0.45, base_grid_resolution=0.15, is_train=True, is_val=False, **kwargs):
        '''
        Zeolite distance grids dataset for 3D CNNs.
        Args: 
            root_dir (string):  root directory of dataset where directory 'distance-grids-h5' exists
            grid_size: grid size as input to the CNN (eg: 100x100x100), is relevant to the grid resolution
            grid_resolution: grid resolution of the input to the CNN (eg: 0.45 Angstrom)
            base_grid_resolution: base grid resolution of the distance grids in the h5py files (eg: 0.15 Angstrom)
            is_train (bool)
            is_val (bool)
        '''
        # IZA-SC:   database of zeolite structures that have been approved by the 
        #           Structure Commission of the International Zeolite Association.
        # PCOD:     predicted crystallography open database.

        dist_folder = kwargs.get('dist_folder', 'distance-grids-h5')              # folder with distance h5py files in a grid resolution of 0.15
        annotations_folder = kwargs.get('annotations_folder', 'C18-adsorption')   # folder with adsorption performance
        annotations_file = kwargs.get('annotations_file', 'each-zeolite-info.csv')  
        train_file = kwargs.get('train_file', 'train_set.txt')                
        val_file = kwargs.get('val_file', 'val_set.txt')
        test_file = kwargs.get('test_file', 'test_set.txt')
       
        # Retrieve filenames and labels
        files = {}
        with open(os.path.join(root_dir, annotations_folder, annotations_file), 'r') as csvfile:
            for line in csv.reader(csvfile):
                if line[0] == 'zeolite': continue    # to ignore header
                if line[2] == '0': continue          # to ignore samples with KH_C18 = 0
                db_set = "IZASC" if line[1] == "IZA" else "PCOD" # set in csv file: {'IZA','PCOD'}
                if not os.path.exists(os.path.join(root_dir, dist_folder, db_set, line[0]+'.h5py')): 
                    continue                                    # to ignore files without dgrid h5py file
                metadict = {'zeolite': line[0],
                            'set': db_set,
                            'kH_C18': np.log(float(line[2]))}  # log applied HERE!
                files[line[0]] = metadict
            
        # split dataset (train/val/test) based on csv/txt files
        self.files = []
        if is_train:            
            with open(os.path.join(root_dir, annotations_folder, train_file), 'r') as csvfile:
                for line in csv.reader(csvfile):
                    if line[0] == 'zeolite': continue    # to ignore header
                    if line[0] in files:
                        self.files.append(files[line[0]])
        elif is_val and not is_train:           # validation set
            with open(os.path.join(root_dir, annotations_folder, val_file), 'r') as csvfile:
                for line in csv.reader(csvfile):
                    if line[0] == 'zeolite': continue    # to ignore header
                    if line[0] in files:
                        self.files.append(files[line[0]]) 
        elif not is_train and not is_val:       # test set
            with open(os.path.join(root_dir, annotations_folder, test_file), 'r') as csvfile:
                for line in csv.reader(csvfile):
                    if line[0] == 'zeolite': continue    # to ignore header
                    if line[0] in files:
                        self.files.append(files[line[0]])

        self.root_dir = root_dir
        self.distance_folder = dist_folder
        self.is_train = is_train
        self.grid_size = grid_size
        self.coord = coordinates_3d(0, grid_size, grid_size, grid_size)
        self.grid_resolution = grid_resolution
        self.base_grid_resolution = base_grid_resolution
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = [self.files[idx]['kH_C18']]                # label (KH-C18)

        distance_name = os.path.join(self.root_dir,
                                     self.distance_folder,
                                     self.files[idx]['set'],
                                     self.files[idx]['zeolite']+'.h5py')
        
        f = h5py.File(distance_name, 'r') # distance grids
        grid = torch.Tensor(np.array(f['distancegrids'])).unsqueeze(0) # the coordinates are in voxel system (a non-orthogonal system)
        _, nx, ny, nz = grid.shape
        bounds = torch.Tensor([nx, ny, nz])

        xvec = f['distancegrids'].attrs['xvec'] # the vector x defining the voxel X-axis
        yvec = f['distancegrids'].attrs['yvec']
        zvec = f['distancegrids'].attrs['zvec']

        x_unit = xvec / (self.grid_resolution / self.base_grid_resolution * np.linalg.norm(xvec)) # the distance grids data is of 0.15 resolution
        y_unit = yvec / (self.grid_resolution / self.base_grid_resolution * np.linalg.norm(yvec))
        z_unit = zvec / (self.grid_resolution / self.base_grid_resolution * np.linalg.norm(zvec))

        A = np.vstack([x_unit, y_unit, z_unit])
        A = torch.Tensor(A)
     
        A_inv = torch.inverse(A) # torch.Size([3, 3]) 

        coord = self.coord.clone() # torch.Size([10000000, 3]) to(device), generate 10000000 coordinates, eg: from (0, 0, 0) to (99, 99, 99)

        if self.is_train:
            coord += torch.rand(3) * bounds # random translation
            
            p = torch.randn(3, 3) # random rotation
            rotation_matrix, _ = torch.linalg.qr(p)
            rand_int = torch.randint(0, 2, (1,)) # 0 or 1
            rotation_matrix = (2*rand_int-1) * rotation_matrix # -1 or 1
            rotation_matrix = torch.transpose(rotation_matrix,0,1) 
            new_coord = torch.mm(coord, rotation_matrix) #torch.Size([1000000, 3])
        else:
            new_coord = coord
            
        grid3d = torch.remainder(new_coord @ A_inv, torch.Tensor([nx, ny, nz])) # coordinates in voxel coordinate system
                
        scalar = torch.diag(2 / bounds) # [-1,1]
        grid3d = torch.add(torch.mm(grid3d, scalar), -1)

        transf_grid = grid3d.reshape(self.grid_size, self.grid_size, self.grid_size, -1)
        grid= grid.permute(0, 3, 2, 1)
        grid = grid[None, :, :, :, :]
        grid_res = F.grid_sample(grid, transf_grid[None, :, :, :, :], align_corners=True) #[1, 1, 100,100,100]
        grid_res = torch.squeeze(grid_res,1) # [1, 100,100,100]

        sample = {
                  'image': grid_res, # [1,100,100,100]
                  'voxel': torch.FloatTensor(A), # [3,3] 
                  'metadata': self.files[idx], # dictionary          
                  'label': torch.FloatTensor(label).float()  # [1]
                  }   
        return sample
