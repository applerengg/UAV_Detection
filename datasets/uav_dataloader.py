import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class UavDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_list = os.listdir(img_dir)
        self.label_list = os.listdir(label_dir)
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        label_path = os.path.join(self.label_dir, self.label_list[idx])
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Load label
        with open(label_path, 'r') as f:
            label = f.read().strip().split()
            label = list(map(float, label))
        
        sample = {"image": image, "label": torch.tensor(label)}
        
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        
        return sample
