import sys
import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
from pytorch3d.ops import marching_cubes, sample_points_from_meshes
from pytorch3d.ops.marching_cubes import marching_cubes
from pytorch3d.structures import Meshes
import dgl
import json

def rotmat_3d(theta, phi, psi):

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    cos_psi = torch.cos(psi)
    sin_psi = torch.sin(psi)

    xmat = torch.eye(3)
    xmat[1, 1] = cos_theta
    xmat[1, 2] = -sin_theta
    xmat[2, 1] = sin_theta
    xmat[2, 2] = cos_theta

    ymat = torch.eye(3)
    ymat[0, 0] = cos_phi
    ymat[0, 2] = sin_phi
    ymat[2, 0] = -sin_phi
    ymat[2, 2] = cos_phi

    zmat = torch.eye(3)
    zmat[0, 0] = cos_psi
    zmat[0, 1] = -sin_psi
    zmat[1, 0] = sin_psi
    zmat[1, 1] = cos_psi

    out = zmat.mm(ymat).mm(xmat)
    return out

def grid_resize(grid, size=100):
    '''
    resize grid to [size, size, size] by repeating grid atomic pattern.
    '''
    H, W, D = grid.shape
    rf = int(np.ceil(max(size/H, size/W, size/D))) # resize factor of smallest dimension size
    enlarged_grid = np.tile(grid, (rf,rf,rf)) # enlarge grid so the smallest dimension is >= size
    return enlarged_grid[:size,:size,:size] 

def coordinates_3d(start=0, H=100, W=100, D=100):
    xs = torch.linspace(start, end=H-1, steps=H)
    ys = torch.linspace(start, end=W-1, steps=W)
    zs = torch.linspace(start, end=D-1, steps=D)
    xc = xs.repeat(W*D, 1).transpose(0, 1).contiguous().view(-1)
    yc = ys.repeat(D, 1).transpose(0, 1).contiguous().view(-1) \
        .repeat(H, 1).contiguous().view(-1)
    zc = zs.repeat(H*W)
    new_coord = torch.cat((xc.unsqueeze(1), yc.unsqueeze(1), zc.unsqueeze(1)), 1)
    return new_coord

def AtomCoordsSampler(src_cif, num_points=1024, cell_size=45, is_train=True):
    structure = Structure.from_file(src_cif)
    super_structure = CubicSupercellTransformation(min_atoms=num_points, min_length=cell_size, force_diagonal=True).apply_transformation(structure)
    atom_coords = super_structure.cart_coords #（N,3)
    atom_coords = torch.from_numpy(atom_coords).float()
    atom_coords = atom_coords[torch.all(atom_coords <= cell_size, axis=1)]
    num_coords = atom_coords.shape[0]
    rf = int(np.ceil(num_points/num_coords))
    atom_coords = torch.tile(atom_coords, (rf,1))
    atom_coords = atom_coords[:num_points,:] #（N,3)
    if is_train:
        rand_indx = torch.randperm(len(atom_coords))
        atom_coords = atom_coords[rand_indx]
    atom_coords = atom_coords.transpose(0, 1) # (3,N)
    return atom_coords # (3,N)

def Uniform_PointSampler(grid, num_points=1024, is_train=True):
    '''
    grid: torch.Size([1,H,W,D])

    '''
    verts, faces = marching_cubes(grid, isolevel=0, return_local_coords=False) 
    verts, faces = verts[0], faces[0]
    verts, faces = verts.unsqueeze(0), faces.unsqueeze(0)
    mesh = Meshes(verts, faces)
    sampled_points = sample_points_from_meshes(mesh, num_points)
    sampled_points = sampled_points.squeeze(0) # [N, 3]
    if is_train:
        rand_indx = torch.randperm(len(sampled_points))
        sampled_points = sampled_points[rand_indx]
    sampled_points = sampled_points.transpose(0, 1)
    return sampled_points # [3,N]

# For zeolite crystal graph
class AtomInitializer(object):
    """
    Base class for initializing the vector representation for atoms.
    
    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        """
        Get embedding of atom_type.
        Returns a 1-d array of length self.embedding_length
        """
        assert atom_type in self.atom_types, 'Invalid atom type: {}'.format(atom_type)
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}
        
    def state_dict(self):
        return self._embedding
    
    def decoder(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]

class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a json file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
      The path to the .json file.
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: Angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
            Minimum interatomic distance
        dmax: float
            Maximum interatomic distance
        step: float
            Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian distance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
            A distance matrix of any shape
        
        Returns
        -------
        expanded_distance: shape (n+1)-d array
            Expanded distance matrix with the last dimension of length
            len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                        self.var**2)

def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
        (atom_fea, nbr_fea, nbr_fea_idx, target)

        atom_fea: torch.Tensor shape (n_i, atom_fea_len)
        nbr_feat: torch.Tensor shape (n_i, M, nbr_fea_len)
        nbr_fea_idx: torch.LongTensor shape (n_i, M)
        target: torch.Tensor shape (1, )
        cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
        Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, sample in enumerate(dataset_list):
        atom_fea, nbr_fea, nbr_fea_idx, target, cif_id = \
            sample['AtomFeatures'], sample['NeighborFeatures'], \
            sample['NeighborIndices'], sample['label'], sample['metadata']['zeolite']
        n_i = atom_fea.shape[0] # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx) # index of neighbors for this crystal
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx) # index of atoms for this crystal
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
            torch.stack(batch_target, dim=0),\
            batch_cif_ids

# For MEGNet and M3GNet
def collate_fn(batch, include_line_graph=False):
    """Merge a list of dgl graphs to form a batch."""
    batch_graph, batch_lattice, batch_state_attr, batch_labels, batch_cif_ids = [], [], [], [], []
    if include_line_graph:
        batch_line_graph = []
        for _, sample in enumerate(batch):
            graph, lattice, state_attr, label, line_graph, cif_id = \
                sample['Graph'], sample['Lattice'], sample['StateAttribute'], sample['label'], sample['LineGraph'], sample['metadata']['zeolite']
            batch_graph.append(graph)
            batch_lattice.append(lattice)
            batch_state_attr.append(state_attr)
            batch_labels.append(label)
            batch_line_graph.append(line_graph)
            batch_cif_ids.append(cif_id)
        g = dgl.batch(batch_graph)
        labels = torch.stack(batch_labels, dim=0)
        state_attrs = torch.stack(batch_state_attr)
        lat = lattice[0] if g.batch_size == 1 else torch.squeeze(torch.stack(batch_lattice))
        l_g = dgl.batch(batch_line_graph)
        return g, lat, labels, state_attrs, l_g, batch_cif_ids
    else:
        for _, sample in enumerate(batch):
            graph, lattice, state_attr, label, cif_id = \
                sample['Graph'], sample['Lattice'], sample['StateAttribute'], sample['label'], sample['metadata']['zeolite']
            batch_graph.append(graph)
            batch_lattice.append(lattice)
            batch_state_attr.append(state_attr)
            batch_labels.append(label)
            batch_cif_ids.append(cif_id)
        g = dgl.batch(batch_graph)
        labels = torch.stack(batch_labels, dim=0)
        state_attrs = torch.stack(batch_state_attr)
        lat = lattice[0] if g.batch_size == 1 else torch.squeeze(torch.stack(batch_lattice))
        return g, lat, labels, state_attrs, batch_cif_ids
    
# for MACE
def compute_avg_num_neighbors(data_loader):
    num_neighbors = []
    for batch in data_loader:
        _, recievers = batch['AtomicData'].edge_index
        _, counts = torch.unique(recievers, return_counts=True)
        num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item()

class Logger(object):
    def __init__(self, output):
        self.terminal = sys.stdout
        self.log = open(output, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
