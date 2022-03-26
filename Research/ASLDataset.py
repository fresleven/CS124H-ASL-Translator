import json
from torchvision.io import read_video
import torch 
from torch import nn 
from torch.utils.data import Dataset
import requests
import glob
import moviepy.editor as moviepy
import os


class CustomImageDataset(Dataset):
    def __init__(self, data_path):

        data = json.load(open('dataset.json'))
        self.json = data
        self.files = set()
        self.data_path = data_path
        for file in glob.glob(data_path+"/*"):
            self.files.add(file.split("\\")[-1].split(".")[0])
        self.samples = []
        urls = set()
        # build OHE LUT
        word_class_idx_LUT = {}
        for i,word in enumerate(self.json):
            word_class_idx_LUT[word] =  i

        for word, data in self.json.items():
           #  print(word, len(data))
            for clip in data:
                start_frame, end_frame, url = clip
                fname = "_".join(url.split("/")[-2:])
                class_idx = word_class_idx_LUT[word]
                self.samples.append((class_idx, start_frame, end_frame, fname))
                self.download_if_missing(fname, url)
                urls.add(url)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        class_idx, start_frame, end_frame, fname = idx
        frames, _, _ = read_video(self.data_path + "\\" + fname)
        frames = frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        frames = frames[start_frame:end_frame]
        return frames, class_idx
    
    def download_if_missing(self, fname, url):
       # print(fname, self.files)
        fname_root = fname.split(".")[0]
        if fname_root in self.files:
            print("Found", fname)
            return
        else:
            print("Downloading", fname)
            r = requests.get(url)
            new_fpath = self.data_path + "/" + fname
            new_fpath_root = self.data_path + "/" + fname_root
            with open(new_fpath, "wb") as file:
                file.write(r.content)
            clip = moviepy.VideoFileClip(new_fpath)
            clip.write_videofile(new_fpath_root+".mp4", audio=False, preset="veryslow", threads=8, logger="bar")
            os.remove(new_fpath)
            
        
if __name__ == '__main__':
    ds = CustomImageDataset("videos")
    print("Found", len(ds), "elements in Dataset!")
    