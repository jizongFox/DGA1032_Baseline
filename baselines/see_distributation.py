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
torch.set_num_threads(1)
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
distribution = {}
'''
for i, (img, mask, weak_mask, path) in tqdm(enumerate(train_loader)):
    path = os.path.basename(path[0])
    slice_number = path.split('_')[2].split('.png')[0]
    if distribution.get(slice_number)==None:
        distribution[slice_number]=[[],[],[]]
    np_mask = mask.squeeze().numpy().ravel()
    class1_number = len(np.where(np_mask>0.8)[0])
    distribution[slice_number][0].append(class1_number)
    class2_number = len([x for x in np_mask if x>0.5 and x<=0.8])
    distribution[slice_number][1].append(class2_number)
    class3_number = len([x for x in np_mask if x<0.5 and x>=0.2])
    distribution[slice_number][2].append(class3_number)
    pass
print()
import pickle
with open('distribution.pkl','wb') as f:
    pickle.dump(distribution,f)
'''
import pickle,pandas as pd
with open('distribution.pkl','rb') as f:
    distribution= pickle.load(f)

slide_numbers = sorted(distribution.keys())
for slide_number in slide_numbers:
    values = pd.Series(distribution[slide_number][2]).plot.density(label = slide_number)
plt.legend()

print()
# # sizes_series.plot
# # plt.show()
# import seaborn as sns
# sns.set_palette("hls") #设置所有图的颜色，使用hls色彩空间
# sns.distplot(sizes,color="r",bins=100,kde=True)
# plt.show()