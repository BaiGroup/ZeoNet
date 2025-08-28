import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights, ResNet50_Weights

class LinearAdaptor(nn.Module):
    def __init__(self, in_channels=3):
        super(LinearAdaptor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        return out

class MVCNN_resnet50(nn.Module):
    def __init__(self, nclasses=1, num_views=3):
        super(MVCNN_resnet50, self).__init__()
        self.nclasses = nclasses
        self.num_views = num_views
        model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = model.fc
        self.classifier = nn.Linear(2048, self.nclasses)

    def forward(self, x):
        y = self.features(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1])) 
        return self.classifier(torch.max(y,1)[0].view(y.shape[0],-1))

class MVCNN_resnet18(nn.Module):
    def __init__(self, nclasses=1, num_views=1, in_channels=3):
        super(MVCNN_resnet18, self).__init__()

        self.adaptor=LinearAdaptor(in_channels)
        self.nclasses = nclasses
        self.num_views = num_views
        model =  torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = model.fc
        self.classifier = nn.Linear(512, self.nclasses)

    def forward(self, x):
        x = self.adaptor(x)
        y = self.features(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))
        out = self.classifier(torch.max(y,1)[0].view(y.shape[0],-1))
        return out