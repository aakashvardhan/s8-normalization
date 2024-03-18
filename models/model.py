import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import get_config
# import sys
# # add parent directory to path
# sys.path.append('/Users/aakashvardhan/Library/CloudStorage/GoogleDrive-vardhan.aakash1@gmail.com/My Drive/ERA v2/s8-normalization/config.py')


GROUP_SIZE_gn = 2
GROUP_SIZE_ln = 1
config = get_config()
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,norm, kernel_size=(3,3),dropout_value=0, **kwargs):
        super().__init__()
        
        if norm == 'bn':
            self.norm = lambda num_features: nn.BatchNorm2d(num_features)
        elif norm == 'gn':
            self.norm = lambda num_features: nn.GroupNorm(GROUP_SIZE_gn, num_features)
        elif norm == 'ln':
            self.norm = lambda num_features: nn.GroupNorm(GROUP_SIZE_ln, num_features)
        else:
            raise ValueError('Norm type {} not supported'.format(norm))
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,bias=False, **kwargs),
            nn.ReLU(),
            self.norm(out_channels),
            nn.Dropout(dropout_value)
        )
        
    def forward(self, x):
        return self.conv(x)
    
class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=(1,1), **kwargs):
        super().__init__()
        
        self.conv1d = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=False, **kwargs),
        )
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.conv1d(x)
        return self.pool(x)
    
class Net(nn.Module):
    def __init__(self, n_channels=32, dropout_value=config['dropout'],norm=config['norm']):
        super(Net, self).__init__()
        
        # Convolution block 1
        self.conv1 = ConvBlock(3, n_channels // 2, kernel_size=(3,3),norm=norm, padding=1) # output_size = 32, RF = 3
        self.conv2 = ConvBlock(n_channels // 2, n_channels, kernel_size=(3,3),norm=norm, padding=1) # output_size = 32, RF = 5
        
        # Transition block 1
        self.conv3 = TransitionBlock(n_channels, n_channels // 2) # output_size = 16, RF = 6
        
        # Convolution block 2
        self.conv4 = ConvBlock(n_channels // 2, n_channels // 2, kernel_size=(3,3),norm=norm, padding=1) # output_size = 16, RF = 10
        self.conv5 = ConvBlock(n_channels // 2, n_channels, kernel_size=(3,3),norm=norm, padding=1) # output_size = 16, RF = 14
        self.conv6 = ConvBlock(n_channels, n_channels, kernel_size=(3,3),norm=norm, padding=1) # output_size = 16, RF = 18
        
        # Transition block 2
        self.conv7 = TransitionBlock(n_channels, n_channels // 2) # output_size = 8, RF = 20
        
        # Convolution block 3
        self.conv8 = ConvBlock(n_channels // 2, n_channels // 2, kernel_size=(3,3),norm=norm, padding=1) # output_size = 8, RF = 36
        self.conv9 = ConvBlock(n_channels // 2, n_channels, kernel_size=(3,3),norm=norm, padding=1) # output_size = 8, RF = 52
        self.conv10 = ConvBlock(n_channels, n_channels, kernel_size=(3,3),norm=norm, padding=1) # output_size = 8, RF = 68
        
        # Output block
        self.gap = nn.AdaptiveAvgPool2d(1) # output_size = 1, RF = 76
        self.conv11 = nn.Conv2d(n_channels, 10, kernel_size=(1,1), padding=0) # output_size = 1, RF = 76
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.gap(x)
        x = self.conv11(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
