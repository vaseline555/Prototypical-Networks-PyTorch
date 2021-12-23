import torch
import torch.nn as nn
from collections import OrderedDict


class ProtoConvNet(nn.Module):
    def __init__(self):
        super(ProtoConvNet, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64, track_running_stats=False)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64, track_running_stats=False)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64, track_running_stats=False)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=64, track_running_stats=False)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.mp1(self.act(self.bn1(self.conv1(x)))) # requires padding=1 for 1600-dim output
        x = self.mp2(self.act(self.bn2(self.conv2(x))))
        x = self.mp3(self.act(self.bn3(self.conv3(x))))
        x = self.mp4(self.act(self.bn4(self.conv4(x))))
        x = self.flatten(x) # 1600-dim output 
        return x