import torch
import torchvision
from ASLDataset import ASLDataset
from AKModel import AKModel

dataset = ASLDataset("/home/weustis/data/asl/videos")


trainloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                          shuffle=True, num_workers=10, pin_memory=True)

model = AKModel().cuda()
epochs = 10
for e in range(epochs):
    for x, y in trainloader:
        print(x.shape, y.shape)
        break