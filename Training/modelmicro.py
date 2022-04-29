import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(in_chan, out_chan, kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(out_chan)
        self.res_conv = nn.Conv3d(in_chan, out_chan, 1, padding='same')
        
    def forward(self, x):
        tmp = x # C 

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x + self.res_conv(tmp)  # C_2 
        return x
    
class Net(nn.Module):
    def __init__(self, num_words=340, frames=40, size=112):
        super().__init__()
        self.conv1 = ConvBlock(3, 16, kernel_size=7)
        self.conv2 = ConvBlock(16,  32)
        self.conv3 = ConvBlock(32, 64)
        self.conv4 = ConvBlock(64, 128)
        self.conv5 = ConvBlock(128, 128)
        
        self.pool1 = nn.MaxPool3d((1,2,2))
        self.pool2 = nn.MaxPool3d(2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(.49)
        self.preds1 = nn.Linear(128*(frames//16)*(size//32)*(size//32), num_words)
    def forward(self, x):

        x = self.conv1(x)  # 40 20 10 5  3

        x = self.pool2(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool2(x)

        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.pool1(x)
        x = torch.flatten(x, start_dim=1)
        
        x = self.dropout(x)

        x = self.preds1(x)

        return x
    
    

if __name__ == "__main__":
    model = Net().cuda()
    x = torch.rand(1, 3, 40, 112, 112).cuda()
    with torch.no_grad():
        y = model(x)
    summary(model, (3, 40, 112, 112))
