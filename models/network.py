import torch
import torch.nn as nn
import torchvision
from models import layers


class HandNetwork(nn.Module):
    def __init__(self, resnet, clazz):
        super(HandNetwork, self).__init__()
        self.pool_size = 1
        self.resnet = resnet
        
        # out_channels multiple by pool size and multiply by 2
        # multiply by 2 is get from torch cat of AdaptiveAvgPool2d and AdaptiveMaxPool2d
        in_features = self.get_last_layer_out_channels() * self.pool_size*self.pool_size*2
        
        self.resnet.avgpool = nn.Sequential(
            layers.AdaptiveConcatPool2d(self.pool_size),
            layers.Flatten()
        )
        
        self.resnet.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features, in_features//2),
            nn.ReLU(inplace=True),
            
            nn.BatchNorm1d(in_features//2),
            nn.Dropout(0.3),
            nn.Linear(in_features//2, in_features//4),
            nn.ReLU(inplace=True),
            
            nn.BatchNorm1d(in_features//4),
            nn.Dropout(0.2),
            nn.Linear(in_features//4, clazz),
        )
        
    def forward(self, x):
        x = self.resnet(x)
        return x
        
    def get_last_layer_out_channels(self):
        if len(self.resnet.layer4) >=3:
            if type(self.resnet.layer4[2]) == torchvision.models.resnet.BasicBlock:
                return self.resnet.layer4[2].conv2.out_channels
            elif type(self.resnet.layer4[2]) == torchvision.models.resnet.Bottleneck:
                return self.resnet.layer4[2].conv3.out_channels
            else:
                return 0
        else:
            if type(self.resnet.layer4[1]) == torchvision.models.resnet.BasicBlock:
                return self.resnet.layer4[1].conv2.out_channels
            elif type(self.resnet.layer4[1]) == torchvision.models.resnet.Bottleneck:
                return self.resnet.layer4[1].conv3.out_channels
            else:
                return 0
            
    def freeze(self):
        for param in self.resnet.parameters():
            param.require_grad = False
        for param in self.resnet.fc.parameters():
            param.require_grad= True
            
    def unfreeze(self):
        for param in self.resnet.parameters():
            param.require_grad = True