
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import pickle as pkl
import pathlib


class SynthtextDataset(Dataset):
    def __init__(self, path):
        # Check clean dirs if on Gilbreth cluster
        if os.path.exists("/scratch/gilbreth/stevenwh/TextDetection/rawdata/Synthtext/clean_dirs.pkl"):
            self.path = path
            self.files = []
            
            dirs = [d for d in os.listdir(self.path)]
            # Use clean_dirs to determine which directories are incomplete and should be skipped
            with open("/scratch/gilbreth/stevenwh/TextDetection/rawdata/Synthtext/clean_dirs.pkl", 'rb') as f:
                clean_dirs = pkl.load(f)

                # Traverse only the clean directories and get file names of all the .npz files
                for d in dirs:
                    if d not in clean_dirs: continue

                    p = f"{path}/{d}"

                    for filename in os.listdir(p):
                        if filename.endswith('.npz'):
                            self.files.append(f"{p}/{filename}")
        else:
            self.path = pathlib.Path(path);
            if(not self.path.exists()):
                raise ValueError(f"Path ({path}) does not exist!");
            self.files = list(self.path.iterdir());    


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(file, allow_pickle=True)
        
        # Extract the data from the .npz file
        image = data['image']
        groundtruth = data['groundtruth']
        
        
        # Convert arrays to tensors
        image_tensor = torch.from_numpy(image).permute(2, 1, 0).float()  
        # make sure each ground truth is the same dimension for batching!
        gt_tensor = torch.zeros(63, 4) - 1; #fill in no object indices with -1
        gt_tensor[:len(groundtruth)] = torch.from_numpy(groundtruth[...,:4].astype(np.float32));        
        
        return image_tensor, gt_tensor