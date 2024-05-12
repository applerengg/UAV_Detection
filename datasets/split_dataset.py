import glob
import os
from pathlib import Path
import random
import shutil

RATIO_TRAIN = 0.80
RATIO_VALID = 0.10
RATIO_TEST  = 0.10

dataset_folder = "uav_dataset_gokboru"
label_ext      = ".txt" 
image_ext      = ".png" # ".png"

#-- assuming all images and label files are inside the "all" folder together
filenames   = glob.glob(f"{dataset_folder}/all/*{label_ext}")
random.shuffle(filenames)
NUM_IMG     = len(filenames)

train_end   = round(NUM_IMG * RATIO_TRAIN)
val_end     = train_end + round(NUM_IMG * RATIO_VALID)
test_end    = NUM_IMG

for s in ["train", "valid", "test"]:
    os.makedirs(f"{dataset_folder}/{s}/images", exist_ok=True)
    os.makedirs(f"{dataset_folder}/{s}/labels", exist_ok=True)

for i, f_label in enumerate(filenames):
    f_stem  = Path(f_label).stem
    f_dir   = os.path.dirname(f_label)
    f_image = f"{f_dir}/{f_stem}{image_ext}"

    print(f"{i}\n {f_label} \n {f_image}")

    if i <= train_end:
        shutil.copy2(f_image, f"{dataset_folder}/train/images")
        shutil.copy2(f_label, f"{dataset_folder}/train/labels")
    elif i <= val_end:
        shutil.copy2(f_image, f"{dataset_folder}/valid/images")
        shutil.copy2(f_label, f"{dataset_folder}/valid/labels")
    elif i <= test_end:
        shutil.copy2(f_image, f"{dataset_folder}/test/images")
        shutil.copy2(f_label, f"{dataset_folder}/test/labels")

