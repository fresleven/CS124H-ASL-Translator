import torch
from torch.utils.data import DataLoader

from dataset_imgs import ASLDataset
from model import Net as Model
from tqdm import tqdm
from utils import accuracy, seed_everything
seed_everything(42)

import wandb
init = True

# dataset + dataloader
# transforms


dataset = ASLDataset(r"/raid/projects/weustis/data/asl/dataset.json")
train_number = int(len(dataset)*.8)
train_set, test_set = torch.utils.data.random_split(dataset, [train_number, len(dataset)-train_number])

trainloader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=40, pin_memory=True, prefetch_factor=2, persistent_workers=True)
testloader = DataLoader(test_set, batch_size=16, shuffle=True, num_workers=10, pin_memory=True, prefetch_factor=2)


# model
model = Model().cuda()


# loss func and opt
crit = torch.nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(),  lr=0.003, momentum=0.9)

for epoch in range(30):
    # train loop
    model.train()
    train_loss_total = 0
    train_top1_total = 0
    train_top5_total = 0
    
    test_loss_total = 0
    test_top1_total = 0
    test_top5_total = 0
    
    for x,y in tqdm(trainloader):
        opt.zero_grad()
        x = x.cuda()
        y = y.cuda()
        
        pred = model(x)
        loss = crit(pred, y)
        train_loss_total += loss.item()/len(x)

        top1, top5 = accuracy(pred, y)
        train_top1_total += top1.item() * len(x)
        train_top5_total += top5.item() * len(x)
        
        loss.backward()
        opt.step()
       
    model.eval()
    with torch.no_grad():
        
        for x,y in tqdm(testloader):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss = crit(pred, y)
            test_loss_total += loss.item()/len(x)
            top1, top5 = accuracy(pred, y)
            test_top1_total += top1.item() * len(x)
            test_top5_total += top5.item() * len(x)
    if init:
        wandb.init("CS124H-GR6-SP22")
        init = False
    wandb.log({
        "train_loss": train_loss_total,
        "train_top1": train_top1_total/len(train_dataset),
        "train_top5": train_top5_total/len(train_dataset),
        "test_loss": test_loss_total,
        "test_top1": test_top1_total/len(test_dataset),
        "test_top5": test_top5_total/len(test_dataset)
    })
