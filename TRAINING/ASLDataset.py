import json
from torchvision.io import read_video
import torch 
from torch import nn 
from torch.utils.data import Dataset
import requests
import glob
from torchvision import transforms
import os


class ASLDataset(Dataset):
    def __init__(self, data_path, clip_len=30):

        data = json.load(open('/home/weustis/data/asl/dataset.json'))
        self.json = data
        self.files = set()
        self.clip_len = clip_len
        
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize(224)
        ])
        
        self.data_path = data_path
        for file in glob.glob(data_path+"/*"):
            self.files.add(file.split("\\")[-1].split(".")[0])
        self.samples = []
        urls = set()
        # build OHE LUT
        word_class_idx_LUT = {}
        for i,word in enumerate(self.json):
            word_class_idx_LUT[word] =  i
        self.total_clips = 0
        for word, data in self.json.items():
           #  print(word, len(data))
            for clip in data:
                start_frame, end_frame, url = clip
                fname = "_".join(url.split("/")[-2:])
                class_idx = word_class_idx_LUT[word]
                self.samples.append((class_idx, start_frame, end_frame, fname))
                self.total_clips += int(end_frame-start_frame) - clip_len + 1     
        print(f"Found {len(self.samples)} videos and {self.total_clips} clips") 
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        class_idx, start_frame, end_frame, fname = self.samples[idx]
        frames, _, _ = read_video(self.data_path + "/" + fname)
        frames = frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        frames = frames[start_frame:end_frame]
        offset_frame = torch.randint(0, end_frame-start_frame-self.clip_len+1, (1,))
        frames = frames[offset_frame:offset_frame + self.clip_len]
        frames = self.transforms(frames)
        return frames, class_idx
    
        
if __name__ == '__main__':
    ds = ASLDataset("videos")
    print("Found", len(ds), "elements in Dataset!")