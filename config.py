import os
import torch
from transform import train_transforms, test_transforms

def get_config():
    config = {
        'seed': 1,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'batch_size': 128,
        'epochs': 20,
        'lr': 0.01,
        'dropout': 0.1,
        'norm': 'bn',
        'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
        'train_transforms': train_transforms,
        'test_transforms': test_transforms,
    }
    return config