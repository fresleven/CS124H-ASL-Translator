import os
import json
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import torchvideo.transforms as tvt
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
import h5py 
import os


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
            
            self.labels[key] = current_class # one_hot_version # make it the next one_hot
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
        video_path = "/raid/projects/weustis/data/asl/videos/"+ "_".join(video_url.split('/')[-2:]).split('.')[0] + "/"
      #  video_path = r"/raid/projects/weustis/data/asl/videos/ASL_2008_01_11_scene26-camera1/"
        # load the video
        path, dirs, files = next(os.walk(video_path))
        frame_count = len(files)
        
        # c, t, h, w
        # specified frames and size beforehand

        # shorten/length the video
        frame_end = int(frame_end)
        frame_start = int(frame_start)
        t = frame_end - frame_start #time of video 
        
        
        if (t < self.frames): #if shorter
            offset = torch.randint(t-self.frames, 0, (1,)) + 1

        if (t > self.frames): 
            offset = torch.randint(t-self.frames, (1,))
        else:
            offset = 0
        frames = ["%06d" % (x,)+".jpg" for x in range(frame_start+offset, frame_start+offset+self.frames)]
        r = torch.empty(self.frames, 3, 480, 640)
        for idx, p_end in enumerate(frames):
            img_data =  read_image(video_path + p_end)
            r[idx, :, :, :] = img_data
         # resize video
        r = r.permute([1,0,2,3]).unsqueeze(0)
        r = next(self.transforms(r))
       
        return r, label
    
if __name__ == "__main__":

    mydataset = ASLDataset("/raid/projects/weustis/data/asl/dataset.json")
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(mydataset, batch_size=8, shuffle=True, num_workers=40, pin_memory=True, prefetch_factor=1, persistent_workers=True)
    
    for x,y in tqdm(dataloader):
        pass

    # python dataloader.py
