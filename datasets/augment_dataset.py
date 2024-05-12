import glob
import os
from pathlib import Path
import numpy as np
import cv2

import torchvision.transforms as tr
import torchvision.io as tio


APPLY_AUGMENTATION_PROB = 1.0
COLOR_AUG_PROB = 0.75
PERSPECTIVE_AUG_PROB = 0.50
SHEAR_AUG_PROB = 0.50
AFFINE_AUG_PROB = 0.75
CROP_AUG_PROB = 0.25
HFLIP_AUG_PROB = 0.75
VFLIP_AUG_PROB = 0.10
OVERWRITE_EXISTING = True

dataset_folder = "uav_dataset_gokboru"
image_ext      = ".png" # ".jpg"
image_save_funcs = {".png": tio.write_png, ".jpg": tio.write_jpeg, ".jpeg": tio.write_jpeg}

#-- assuming all images are inside the "all" folder 
filenames   = glob.glob(f"{dataset_folder}/all/*{image_ext}")
NUM_IMG     = len(filenames)

os.makedirs(f"{dataset_folder}/augmented", exist_ok=True)

color_augmenter = tr.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.9, hue=0.2)
pers_augmenter = tr.RandomPerspective(distortion_scale=0.2, p=1)
shear_augmenter = tr.RandomAffine(degrees=0, shear=15)
affine_augmenter = tr.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.75, 1.25))
# crop_augmenter = tr.RandomResizedCrop(size=640, scale=(0.1, 0.9), ratio=(0.75, 1.25))
flip_augmenter = tr.Compose([tr.RandomHorizontalFlip(p=HFLIP_AUG_PROB), tr.RandomVerticalFlip(p=VFLIP_AUG_PROB)])

for i, f in enumerate(filenames):
    f_stem  = Path(f).stem
    save_path = f"{dataset_folder}/augmented/{f_stem}_aug{image_ext}"
    if os.path.isfile(save_path) and not OVERWRITE_EXISTING:
        continue
    
    img_tensor = tio.read_image(f, tio.ImageReadMode.RGB).to(device=0) # remove to(device) for cpu tensor
    
    if np.random.rand() < COLOR_AUG_PROB:
        img_tensor = color_augmenter(img_tensor) 
    if np.random.rand() < PERSPECTIVE_AUG_PROB: # at most one is applied among perspective and shear
        img_tensor = pers_augmenter(img_tensor) 
    elif np.random.rand() < SHEAR_AUG_PROB: # at most one is applied among perspective and shear
        img_tensor = shear_augmenter(img_tensor) 
    if np.random.rand() < AFFINE_AUG_PROB:
        img_tensor = affine_augmenter(img_tensor) 
    # img_tensor = crop_augmenter(img_tensor) 
    img_tensor = flip_augmenter(img_tensor) 

    print(i, f)
    # cv2.imshow("img", cv2.cvtColor(img_tensor.permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB) )
    # cv2.waitKey(0)

    image_save_funcs[image_ext](img_tensor.to(device="cpu"), save_path)
