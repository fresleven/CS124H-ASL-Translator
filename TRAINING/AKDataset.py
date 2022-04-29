import os
import json
import torch
from torchvision.io import read_video
from torch.utils.data import Dataset

class AKDataset(Dataset):
    def __init__(self, json_path, frame=60, size=256):
        f = open(json_path)
        data = json.load(f)
        
        self.clips = []
        
        self.labels = {}
        total_classes = len(data)
        current_class = 0
        
        self.frames = frame
        self.size = size

        for key in data:
            
            one_hot_version = torch.nn.functional.one_hot(torch.tensor([current_class]), num_classes=total_classes)
            self.labels[key] = one_hot_version # make it the next one_hot
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
        video_path = "/home/weustis/data/asl/videos/"+ "_".join(video_url.split('/')[-2:]).split('.')[0] + ".mp4"
        print(video_path)
        # load the video
        r = read_video(video_path)
        # slice the video
        r = r[frame_start:frame_end]
        # t, h, w, c
        # c, t, h, w
        r = r.permute([3, 0, 1, 2]) 
        # specified frames and size beforehand
        # resize the video H = 480, W = 640
        hstart = (480 - self.size)/2
        wstart = (640 - self.size)/2
        r = r[:][:][hstart:hstart+self.size][wstart:wstart+self.size]
        # shorten/length the video
        t = r.shape[1] #time of video
        if (t < self.frames): #if shorter, repeats frames
            tmp = torch.full((self.frames, r.shape[2], r.shape[3], r.shape[0]), 0)
            tmp = tmp.permute([3, 0, 1, 2])
            for i in range(self.frames / t):
                tmp[:][i*t:i*t+t] = r
            tmp[:][-(self.frames % t):] = r[:][:(self.frames%t)]
            r = tmp
        if (t > self.frames): # if longer, shortens to middle frames
            timestart = (t - self.frames)/2
            #alternate
            #timestart = torch.randint(0, t - self.frames)
            r = r[:][timestart:timestart+self.frames]
        return r, label
    
    
if __name__ == "__main__":
    ds = AKDataset("/home/weustis/data/asl/dataset.json")
    print(len(ds))
    x, y = ds[0]
    print(x.shape)
    print(y)