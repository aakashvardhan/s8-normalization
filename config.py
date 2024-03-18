import os
import torch

def get_config():
    config = {
        'seed': 1,
        'batch_size': 128,
        'test_batch_size': 1000,
        'epochs': 20,
        'lr': 0.01,
        'dropout': 0.1,
        'norm': 'bn',
        'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    }
    return config