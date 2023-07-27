import sys
import numpy as np
import torch


def rotmat_3d(theta, phi, psi):
    #print(theta, phi, psi)

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

    #xmat = torch.tensor(
    #        [[1., 0., 0.],
    #         [0., cos_theta, -sin_theta],
    #         [0., sin_theta, cos_theta]])

    #ymat = torch.tensor(
    #        [[cos_phi, 0., sin_phi],
    #         [0., 1., 0.],
    #         [-sin_phi, 0., cos_phi]])

    #zmat = torch.tensor(
    #        [[cos_psi, -sin_psi, 0.],
    #         [sin_psi, cos_psi, 0.],
    #         [0., 0., 1.]])

    out = zmat.mm(ymat).mm(xmat)
    #out = xmat
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

class Logger(object):
    def __init__(self, output):
        self.terminal = sys.stdout
        self.log = open(output, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # after python 3, should have
        pass
