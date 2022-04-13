import os
import json
import torch
from torchvision.io import read_video
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import torchvideo.transforms as tvt
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
class ASLDataset(Dataset):
    def __init__(self, json_path, frame=60, size=256, transforms=None):
        f = open(json_path, 'r')
        data = json.load(f)
        
        self.clips = []
        
        self.labels = {}
        total_classes = len(data)
        current_class = 0
        
        self.frames = frame
        self.size = size
        if transforms is None:
            transforms =  Compose([
                tvt.NormalizeVideo([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], channel_dim=1),
                tvt.ResizeVideo((size,size), interpolation=InterpolationMode.BICUBIC)
               
            ])
     
        self.transforms = transforms

        for key in data: # for each word
            one_hot_version = torch.nn.functional.one_hot(torch.tensor([current_class]), num_classes=total_classes)
            
            self.labels[key] = one_hot_version # make it the next one_hot
            # word -> OHE
            current_class += 1
            
            for value in data[key]:
                self.clips.append((key, value))
        
        
    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        word, video_data = self.clips[idx]
        # convert the word to a one hot encoding
        label = self.labels[word]
        
        # convert the video data to be the same shape (and load the video)
        frame_start, frame_end, video_url = video_data
        # get the video path
        video_path = "/raid/projects/weustis/data/asl/videos/"+ "_".join(video_url.split('/')[-2:]).split('.')[0] + ".mp4"
        # load the video
        r = read_video(video_path)
        r = r[0]
        # slice the video

        # t, h, w, c
        # c, t, h, w
        r = r.permute([3, 0, 1, 2]) 
        # specified frames and size beforehand

        # shorten/length the video
        t = frame_end - frame_start #time of video 
        if (t < self.frames): #if shorter
            offset = torch.randint(t-self.frames, 0, (1,)) + 1

        if (t > self.frames): 
            offset = torch.randint(t-self.frames, (1,))
            
        r = r[:, frame_start+offset:frame_start+offset+self.frames]
         # resize video
        r = r.unsqueeze(0).float()
        r = self.transforms(r)
        tmp = None

        for x in r:
            tmp = x
        r = tmp
        return r, label
    
if __name__ == "__main__":

    mydataset = ASLDataset("/raid/projects/weustis/data/asl/dataset.json")
    print(len(mydataset))
    
    for x,y in tqdm(mydataset):
        print(x.shape)
        print(y.shape)

    # python dataloader.py
