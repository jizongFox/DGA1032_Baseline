# coding=utf-8
import copy, os, sys, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import utils.medicalDataLoader as medicalDataLoader
from utils.utils import Colorize, evaluate_iou
sys.path.insert(-1, os.getcwd())
warnings.filterwarnings('ignore')

data_dir = '../dataset/ACDC-2D-All'
batch_size = 1
batch_size_val = 1
num_workers = 1

color_transform = Colorize()
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])
mask_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])

train_set = medicalDataLoader.MedicalImageDataset('train', data_dir, transform=transform, mask_transform=mask_transform,
                                                  augment=True, equalize=False)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_set = medicalDataLoader.MedicalImageDataset('val', data_dir, transform=transform, mask_transform=mask_transform,
                                                equalize=False)
val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)

sizes=[]
for i, (img, mask, weak_mask, _) in tqdm(enumerate(train_loader)):
    if mask.sum()==0:
        continue
    sizes.append(mask.sum().item())

sizes_series = pd.Series(sizes)

# sizes_series.plot
# plt.show()
import seaborn as sns
sns.set_palette("hls") #设置所有图的颜色，使用hls色彩空间
sns.distplot(sizes,color="r",bins=100,kde=True)
plt.show()