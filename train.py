import sys

sys.path.insert(0, '.')
sys.path.insert(0, './datasets')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as Data
import argparse

from model import generate_model
from zeolites import *

import numpy as np

from sklearn.metrics import r2_score
import os
from utils import Logger
import pandas as pd

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a model for Zeolites classification')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--gpu', default='', type=str, help='CUDA visible device')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer as in PyTorch')
    parser.add_argument('--log_interval', type=int, default=1000, help='logging training status')
    parser.add_argument('--save_dir', type=str, default='checkpoints/001/', help='for checkpoints')
    parser.add_argument('--model', type=str, default='resnet', help='select which model')
    parser.add_argument('--model_hp', type=int, default=18, help='select model hyperparameters like model depth/input dimension')
    parser.add_argument('--dset_root', type=str, default='.', help='root directory of dataset')
    parser.add_argument('--grid_resolution', type=float, default=0.45, help='select which grid resolution')
    parser.add_argument('--grid_size', type=int, default=100, help='select which grid size')
    args = parser.parse_args()
    return args

def train(epoch, train_loader):
    net.train()
    train_loss = 0
    num_data = 0
    for batch_idx, sample in enumerate(train_loader):
        data, target = sample['image'], sample['label']
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        num_data += len(data)
        optimizer.zero_grad()
        output = net(data)
        loss = loss_func(output, target)
        train_loss += loss.data
        loss.backward()
        optimizer.step()
        r2_batch = r2_score(target.cpu().detach().numpy(), output.cpu().detach().numpy())
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {:2d} [{:4d}/{} ({:3.0f}%)] MSELoss: {:.4f} ({:.3f}) R2: {:.3f} lr: {:.0e} '.format(
        #         epoch, batch_idx*len(data), len(train_loader.dataset),
        #         100.*batch_idx / len(train_loader), loss.data, train_loss/(batch_idx+1.0),
        #         r2_batch, optimizer.param_groups[-1]['lr']))
    return train_loss/(batch_idx+1.0), r2_batch

def val(epoch, val_loader, show=True):
    net.eval()
    num_data = 0
    zeolites = np.array([], dtype=object).reshape(0)
    scores = np.array([], dtype=np.float32).reshape(0,1) 
    targets = np.array([], dtype=np.int64).reshape(0,1)
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            data, target= sample['image'], sample['label']
            zeolite = sample['metadata']['zeolite']
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            num_data += len(data)
            output = net(data)
            scores = np.concatenate((scores, output.data.cpu().numpy()))
            targets = np.concatenate((targets, target.data.cpu().numpy()))
            zeolites = np.concatenate((zeolites, zeolite))
    mse = loss_func(torch.Tensor(scores), torch.Tensor(targets))
    r2 = r2_score(targets, scores)
    if show:
        print('val R2: {:.2f} (MSE: {:.2f}) '.format(r2, mse))
    return r2, targets, scores, mse, zeolites

if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # log file
    output_file = os.path.join(args.save_dir,'output.log')
    sys.stdout = Logger(output_file) # This line is saved in output.log and STDOUT

    # set CUDA and seed
    args.cuda = args.cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # initialize model
    net = generate_model(model=args.model, model_hp=args.model_hp)

    # load a pre-trained model
    # net.load_state_dict(torch.load('./checkpoints/001/model.pth'))

    if args.cuda:
        net.cuda()

    # initialize optimizer
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
    elif args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr)
    else:
        print('optimizer not defined')

    loss_func = torch.nn.MSELoss()  # this is for mean squared loss
    
    print('[loading datasets...]')
    trainset = ZeoDistGridsDataset(root_dir=args.dset_root, grid_size=args.grid_size, grid_resolution=args.grid_resolution, is_train=True, is_val=False)
    train_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    valset = ZeoDistGridsDataset(root_dir=args.dset_root, grid_size=args.grid_size, grid_resolution=args.grid_resolution, is_train=False, is_val=True)
    val_loader = Data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    print('done.')
    print('training samples: %d, validation samples: %d'%(len(train_loader.dataset), len(val_loader.dataset)))

    try:
        r2, targets, val_scores, mse, zeolites = val(0, val_loader, show=False)
        print('> validation before training R2 %.3f MSE %.3f'%(r2, mse))
        print('[training for %d epochs...]'%args.epochs)
        best_r2 = -np.inf
        for epoch in range(1, args.epochs + 1):
            train_loss, train_r2 = train(epoch, train_loader)
            print('> training after %d epochs R2 %.3f MSE %.3f'%(epoch, train_r2, train_loss))
            r2, targets, val_scores, mse, zeolites = val(epoch, val_loader, show=False)
            print('> validation after %d epochs R2 %.3f MSE %.3f'%(epoch, r2, mse))
            if r2 > best_r2:
                torch.save(net.state_dict(), '%s/best_model.pth' % (args.save_dir)) 
                df = pd.DataFrame({'zeolite' : zeolites, 'target' : np.squeeze(targets), 'prediction' : np.squeeze(val_scores)})
                df.to_csv(os.path.join(args.save_dir,'best_prediction.csv'), index=False)
                best_r2 = r2
            torch.save(net.state_dict(), '%s/model.pth' % (args.save_dir))
    except KeyboardInterrupt:
        print('Exiting from training early')

