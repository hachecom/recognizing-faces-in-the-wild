import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class DeepConv(nn.Module):
    def __init__(self):
        super(DeepConv, self).__init__()

