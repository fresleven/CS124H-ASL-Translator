import h5py
import json
import os
import json
import torch
from torchvision.io import read_video
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import torchvideo.transforms as tvt
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
import glob
from dataset import ASLDataset
json_path = "/raid/projects/weustis/data/asl/dataset.json"
size = 256
transforms =  Compose([
            tvt.NormalizeVideo([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], channel_dim=1),
            tvt.ResizeVideo((256,256), interpolation=InterpolationMode.BICUBIC)
               
        ])
for video in tqdm(glob.glob(r"/raid/projects/weustis/data/asl/videos/ASL_2006_09_26_scene61-camera*")):
    new_path = video.split(".")[0] + ".h5"
    video = read_video(video)[0].unsqueeze(0).permute([0, 4, 1, 2, 3]).float()
    
    video = next(transforms(video))
    #print(video.shape)
    hf = h5py.File(new_path, 'w')
    hf.create_dataset('video', data=video)
    hf.close()
    print("Saved", new_path)
