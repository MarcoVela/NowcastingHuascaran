import torch.utils.data
import os
import sys
sys.path.append(os.path.dirname(__file__))
from MovingMNIST import MovingMNIST

train_set = MovingMNIST(root='../../data/mnist', train=True, download=True)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=100,
                 shuffle=True)