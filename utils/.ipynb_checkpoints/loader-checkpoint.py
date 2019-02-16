import os
import torch
import torch.nn as nn
import torchvision
from models import network

# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_network(filename):
    if os.path.isfile(filename): 
        checkpoint = torch.load(filename, map_location='cpu')
        resnet = torchvision.models.resnet18(pretrained=True)
        clazz = checkpoint['total_clazz']
        model = network.HandNetwork(resnet, clazz)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    else:
        return None
    

def load_checkpoint(filename):
    if os.path.isfile(filename): 
        checkpoint = torch.load(filename, map_location='cpu')
        return checkpoint
    else:
        return None