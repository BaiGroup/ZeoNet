# 3D AlexNet in Pytorch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, input_dim=100):
        super(AlexNet, self).__init__()
        self.feats = 16
        self.feats2 = 128
        self.input_dim = input_dim
        self.conv1 = nn.Conv3d(1, self.feats, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(self.feats)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(self.feats, self.feats, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(self.feats)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)   
        self.conv4 = nn.Conv3d(self.feats, self.feats, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(self.feats)
        self.relu4 = nn.ReLU()
        self.conv6 = nn.Conv3d(self.feats, self.feats, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm3d(self.feats)
        self.relu6 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv8 = nn.Conv3d(self.feats, self.feats, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm3d(self.feats)
        self.relu8 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.feats, self.feats, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm3d(self.feats)
        self.relu10 = nn.ReLU()
        self.conv12 = nn.Conv3d(self.feats, self.feats, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm3d(self.feats)
        self.relu12 = nn.ReLU()         
        self.fc1 = nn.Linear(self.feats*(self.input_dim // 4)**3, self.feats2)
        self.fc2 = nn.Linear(self.feats2, 1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.pool2(x)
        x = self.relu8(self.bn8(self.conv8(x)))
        x = self.relu10(self.bn10(self.conv10(x)))
        x = self.relu12(self.bn12(self.conv12(x)))
        x = x.view(-1, self.feats*(self.input_dim  // 4)**3)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
