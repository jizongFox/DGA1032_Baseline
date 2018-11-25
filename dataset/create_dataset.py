# coding=utf-8
import copy, os, sys, pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.getcwd()))
import matplotlib.pyplot as plt
import warnings
import torch
import sqlite3
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
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
val_set = medicalDataLoader.MedicalImageDataset('val', data_dir, transform=transform, mask_transform=mask_transform,
                                                equalize=False)
val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)

sizes=[]
distribution = {}
db = sqlite3.connect('size_distribution')
create_table_script = '''
CREATE TABLE IF NOT EXISTS sizeinfo(id INTEGER PRIMARY KEY, patient_number TEXT, scan_number TEXT, slice_number TEXT, class1 INTEGER, class2 INTEGER, class3 INTEGER)
'''
try:
    db.execute(create_table_script)
    db.commit()
except Exception as e:
    print(e)
cursor = db.cursor()
for i, (img, mask, weak_mask, path) in tqdm(enumerate(train_loader)):
    path = os.path.basename(path[0])
    patient_number = path.split('patient')[1].split('_')[0]
    scan_number = path.split('patient')[1].split('_')[1]
    slice_number = '%.2d'%int(path.split('_')[2].split('.png')[0])
    np_mask = mask.squeeze().numpy().ravel()
    class1_number = len(np.where(np_mask>0.8)[0])
    class2_number = len([x for x in np_mask if x>0.5 and x<=0.8])
    class3_number = len([x for x in np_mask if x<0.5 and x>=0.2])
    cursor.execute('''INSERT INTO sizeinfo(patient_number, scan_number, slice_number, class1,class2,class3)
                      VALUES(?,?,?,?,?,?)''', (patient_number, scan_number, slice_number, class1_number,class2_number,class3_number))
    db.commit()
db.close()
