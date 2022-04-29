
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary



class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(in_chan, out_chan, kernel_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(out_chan)
        self.res_conv = nn.Conv3d(in_chan, out_chan, 1)
        
    def foward(self, x):
        tmp = x # C 
        tmp = self.res_conv(tmp)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x + tmp  # C_2 
        return x