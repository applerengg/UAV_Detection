import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from uav_dataloader import *

# Sample transform operation
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

IMG_DIR = r'/mnt/c/Users/alpas/OneDrive/ITU-DERSLER/Deep Learning/Project/UAV_Detection/datasets/sim_dataset/train/images'
LABEL_DIR = r'/mnt/c/Users/alpas/OneDrive/ITU-DERSLER/Deep Learning/Project/UAV_Detection/datasets/sim_dataset/train/labels'

dataset = UavDataset(img_dir=IMG_DIR, label_dir=LABEL_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

def show_images_with_labels(dataloader):
    batch = next(iter(dataloader))
    images = batch["image"]
    labels = batch["label"]
    
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    
    for i in range(len(images)):
        image = transforms.ToPILImage()(images[i])
        draw = ImageDraw.Draw(image)
        
        label = labels[i].numpy()
        # Örnek: [class, x_center, y_center, width, height]
        class_id, x_center, y_center, width, height = label
        x_center *= image.width
        y_center *= image.height
        width *= image.width
        height *= image.height
        
        left = x_center - width / 2
        top = y_center - height / 2
        right = x_center + width / 2
        bottom = y_center + height / 2
        
        draw.rectangle([left, top, right, bottom], outline="red", width=2)
        draw.text((left, top), f'Class: {int(class_id)}', fill="red")
        
        axes[i].imshow(image)
        axes[i].axis('off')
    
    plt.show()

# Visualize
show_images_with_labels(dataloader)
