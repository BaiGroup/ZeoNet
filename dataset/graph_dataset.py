import os
import numpy as np
import warnings

from utils.data_utils import AtomCustomJSONInitializer, GaussianDistance

import torch
from torch.utils.data import Dataset
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset

import csv
import warnings

import matgl
from matgl.graph.compute import compute_pair_vector_and_distance, create_line_graph
from matgl.ext.pymatgen import Structure2Graph
from matgl.config import DEFAULT_ELEMENTS

from pymatgen.core.structure import Structure
from torch.utils.data import Dataset
import csv

import numpy as np
from torch.utils.data import Dataset
import csv
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import ase.data

from mace.data.atomic_data import AtomicData
from mace.tools.utils import atomic_numbers_to_indices, get_atomic_number_table_from_zs
from mace.data.neighborhood import get_neighborhood
from mace.tools.torch_tools import to_one_hot

class ZeoCGCNNGraphDataset(Dataset):
    ''' Zeolites dataset for CGCNN'''

    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2, is_train=True, is_val=False, **kwargs):

        cif_folder = kwargs.get('cif_folder', 'CIFs')              # folder with CIF files
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
                if not os.path.exists(os.path.join(root_dir, cif_folder, db_set, line[0]+'.cif')): 
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
        self.cif_folder = cif_folder
        self.is_train = is_train
        
        # for atom features
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        atom_init_file = os.path.join(root_dir, annotations_folder, 'atom_init.json')
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = [self.files[idx]['kH_C18']]                # label (KH-C18)

        cif_name = os.path.join(self.root_dir,
                                self.cif_folder,
                                self.files[idx]['set'],
                                self.files[idx]['zeolite']+'.cif')
        
        crystal = Structure.from_file(cif_name)
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs] # sort by distance
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph.'
                              'If it happens frequently, consider increase '
                              'radius.'.format(self.files[idx]['zeolite']))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + 
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) + 
                               [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)

        sample = {
                  'AtomFeatures': torch.Tensor(atom_fea),
                  'NeighborFeatures': torch.Tensor(nbr_fea),
                  'NeighborIndices': torch.LongTensor(nbr_fea_idx),
                  'metadata': self.files[idx], # dictionary          
                  'label': torch.FloatTensor(label).float()  # [1]              
                  }

        return sample
    
class ZeoMEGNETGraphDataset(Dataset):
    ''' Zeolites dataset for MEGNet and M3GNet'''

    def __init__(
        self, 
        root_dir,
        cutoff=4.0,
        threebody_cutoff=4.0,
        include_line_graph=False,
        directed_line_graph=False,
        is_train=True, 
        is_val=False,
        **kwargs):

        cif_folder = kwargs.get('cif_folder', 'CIFs')              # folder with CIF files
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
                if line[2] == '0': continue          # to ignore samples with KH_C18 = 0 (?)
                db_set = "IZASC" if line[1] == "IZA" else "PCOD" # set in csv file: {'IZA','PCOD'}
                if not os.path.exists(os.path.join(root_dir, cif_folder, db_set, line[0]+'.cif')): 
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
        self.cif_folder = cif_folder
        self.is_train = is_train
        
        # for atom features
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.converter = Structure2Graph(element_types=DEFAULT_ELEMENTS, cutoff=self.cutoff)
        self.include_line_graph = include_line_graph
        self.directed_line_graph = directed_line_graph

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = [self.files[idx]['kH_C18']]                # label (KH-C18)

        cif_name = os.path.join(self.root_dir,
                                self.cif_folder,
                                self.files[idx]['set'],
                                self.files[idx]['zeolite']+'.cif')
        
        structure = Structure.from_file(cif_name)
        graph, lattice, state_attr = self.converter.get_graph(structure)
        graph.ndata["pos"] = torch.tensor(structure.cart_coords)
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lattice[0])
        bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
        graph.edata["bond_vec"] = bond_vec
        graph.edata["bond_dist"] = bond_dist
        if self.include_line_graph:
            line_graph = create_line_graph(graph, self.threebody_cutoff, directed=self.directed_line_graph)
            for name in ["bond_vec", "bond_dist", "pbc_offset"]:
                line_graph.ndata.pop(name)
        graph.ndata.pop("pos")
        graph.edata.pop("pbc_offshift")
        state_attr = torch.tensor(np.array(state_attr), dtype=matgl.float_th)

        if self.include_line_graph:
            sample = {
                "Graph": graph,
                "Lattice": lattice,
                "StateAttribute": state_attr,
                "LineGraph": line_graph,
                "metadata": self.files[idx],
                "label": torch.FloatTensor(label).float(),
            }
        else:
            sample = {
                "Graph": graph,
                "Lattice": lattice,
                "StateAttribute": state_attr,
                "metadata": self.files[idx],
                "label": torch.FloatTensor(label).float(),
            }
        return sample
    
class ZeoMACEGraphDataset(Dataset):
    ''' Zeolites dataset for MACE model.'''

    def __init__(self, root_dir, cutoff=4.0, is_train=True, is_val=False, **kwargs):

        cif_folder = kwargs.get('cif_folder', 'CIFs')                             # folder with CIF files
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
                if not os.path.exists(os.path.join(root_dir, cif_folder, db_set, line[0]+'.cif')): 
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
        self.cif_folder = cif_folder
        self.is_train = is_train
        self.cutoff = cutoff
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = [self.files[idx]['kH_C18']]                # label (KH-C18)

        cif_name = os.path.join(self.root_dir,
                                self.cif_folder,
                                self.files[idx]['set'],
                                self.files[idx]['zeolite']+'.cif')
        
        structure = Structure.from_file(cif_name)
        atoms = AseAtomsAdaptor().get_atoms(structure)
        atomic_numbers = np.array([ase.data.atomic_numbers[symbol] for symbol in atoms.symbols])
        positions = atoms.get_positions()
        pbc = tuple(atoms.get_pbc())
        cell = np.array(atoms.get_cell())
        z_table = get_atomic_number_table_from_zs(atomic_numbers)
        edge_index, shifts, unit_shifts = get_neighborhood(
            positions = positions,
            cutoff = self.cutoff,
            pbc = pbc,
            cell = cell
        ) # edge_index, shifts, unit_shifts, _ for mace v 0.3.14
        indices = atomic_numbers_to_indices(atomic_numbers, z_table)
        one_hot = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )
        cell = torch.tensor(cell, dtype=torch.get_default_dtype())
        atomic_energies_dict = {8:-1.88459, 14:-0.81241735}
        atomic_energies = np.array([atomic_energies_dict[z] for z in z_table.zs])
        
        atomicdata = AtomicData(edge_index=torch.tensor(edge_index, dtype=torch.long),
                                positions=torch.tensor(positions, dtype=torch.get_default_dtype()),
                                shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
                                unit_shifts=torch.tensor(unit_shifts, dtype=torch.get_default_dtype()),
                                cell=cell,
                                node_attrs=one_hot,
                                weight=None,
                                energy_weight=None,
                                forces_weight=None,
                                stress_weight=None,
                                virials_weight=None,
                                forces=None,
                                energy=None,
                                stress=None,
                                virials=None,
                                dipole=None,
                                charges=None
                                ) 

        sample = {
            "AtomicData": atomicdata,
            "E0s": torch.tensor(atomic_energies),
            "zeolite": self.files[idx]['zeolite'],
            "label": torch.FloatTensor(label).float(),
        }

        return sample