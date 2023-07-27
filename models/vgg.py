# 3D VGG11/13/16/19 in Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self, depth=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']):
        super(VGG, self).__init__()
        self.features = self._make_layers(depth)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, depth):
        layers = []
        in_channels = 1
        for x in depth:
            if x == 'M':
                layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv3d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm3d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

def generate_model(model_depth, **kwargs):
    
    cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512], # remove the last pooling layer 'M'
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    assert model_depth in [11, 13, 16, 19]

    if model_depth == 11:
        model = VGG(cfg['VGG11'], **kwargs)
    elif model_depth == 13:
        model = VGG(cfg['VGG13'], **kwargs)  
    elif model_depth == 16:
        model = VGG(cfg['VGG16'], **kwargs)
    elif model_depth == 19:
        model = VGG(cfg['VGG19'], **kwargs)

    return model

