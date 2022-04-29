import torch
import torch.nn as nn
import torch.nn.functional as F
#Tools Conv3D, MaxPool3D, relu, flatten
# 3, frames, size, size = C T H W
#possible changes = use pool1 more often instead of pool2
#add other layers
from torchsummary import summary

class AKModel(nn.Module):
    def __init__(self, num_words=340, frames=30,size=224):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, 3, padding = 'same')
        self.conv2 = nn.Conv3d(32, 32, 3, padding = 'same')
        self.conv3 = nn.Conv3d(32, 32, 3, padding = 'same')
        self.conv4 = nn.Conv3d(32, 32, 3, padding = 'same')
        self.conv5 = nn.Conv3d(32, 64, 3, padding = 'same') 

        self.conv6 = nn.Conv3d(64, 64, 3, padding = 'same')
        self.conv7 = nn.Conv3d(64, 64, 3, padding = 'same')

        self.conv8 = nn.Conv3d(64, 128, 3, padding = 'same')
        self.conv9 = nn.Conv3d(128, 128, 3, padding = 'same')
        self.conv10 = nn.Conv3d(128, 128, 3, padding = 'same')

        self.conv11 = nn.Conv3d(128, 128, 3, padding = 'same')
        self.conv12 = nn.Conv3d(128, 128, 3, padding = 'same')
        self.conv13 = nn.Conv3d(128, 128, 3, padding = 'same') # could go to 256/512 somewhere here
        self.conv14 = nn.Conv3d(128, 128, 3, padding = 'same')
        self.conv15 = nn.Conv3d(128, 256, 3, padding = 'same')
        self.conv16 = nn.Conv3d(256, 256, 3, padding = 'same')
        self.conv17 = nn.Conv3d(256, 256, 3, padding = 'same')
        self.conv18 = nn.Conv3d(256, 256, 3, padding = 'same')
        self.conv19 = nn.Conv3d(256, 256, 3, padding = 'same')
        self.conv20 = nn.Conv3d(256, 256, 3, padding = 'same')
        self.conv21 = nn.Conv3d(256, 256, 3, padding = 'same')
        self.conv22 = nn.Conv3d(256, 256, 3, padding = 'same')

        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.pool2 = nn.MaxPool3d(2)
        self.relu = nn.ReLU()
        self.preds1 = nn.Linear(34560, num_words)

    def forward(self, x):
        #C T H W = 3 self.frames self.size self.size
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        tmp = x
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = tmp + x
        x = self.pool1(self.relu(self.conv5(x))) # 64 self.frames self.size/2 self.size/2
        tmp = x
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = tmp + x 
        x = self.relu(self.conv8(x))
        tmp = x 
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x)) 
        x = tmp + x 
        x = self.pool1(x) # C T H W = 128 self.frames self.size/4 self.size/4
        tmp = x
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = tmp + x
        x = self.pool1(x) # 128 self.frames self.size/8 self.size/8
        tmp = x
        x = self.relu(self.conv13(x)) 
        x = self.relu(self.conv14(x)) 
        x = tmp + x
        x = self.pool1(self.relu(self.conv15(x))) #256 self.frames self.size/16 self.size/16
        tmp = x
        x = self.relu(self.conv16(x)) 
        x = self.relu(self.conv17(x)) 
        x = tmp + x 
        x = self.pool1(x) # 256 self.frames self.size/32 self.size/32
        tmp = x
        x = self.relu(self.conv18(x)) 
        x = self.relu(self.conv19(x))
        x = tmp + x
        x = self.pool2(x) # 256 sel.frames/2 self.size/64 self.size/64
        tmp = x
        x = self.relu(self.conv20(x)) 
        x = self.relu(self.conv21(x))
        x = tmp + x
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        print(x.shape)
        x = self.relu(self.preds1(x))
        return x
    
if __name__ == "__main__":
    model = AKModel().cuda()
    summary(model, input_size=(3, 30, 224, 224))