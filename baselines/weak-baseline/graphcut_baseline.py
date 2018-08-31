import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import os
import sys
from torchnet.meter import AverageValueMeter
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
import warnings
warnings.filterwarnings('ignore')
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import utils.medicalDataLoader as medicalDataLoader
from utils.utils import Colorize, evaluate_iou,graphcut_with_FG_seed_and_BG_dlation

from tqdm import tqdm
import click

filename = os.path.basename(__file__).split('.')[0]
data_dir = '../../dataset/ACDC-2D-All'

batch_size =1
batch_size_val =1
num_workers =1

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

db_meter = AverageValueMeter()
df_meter = AverageValueMeter()

@click.command()
@click.option('--kernel_size',default=5)
@click.option('--lamda',default=1.0)
@click.option('--sigma',default=0.01)
@click.option('--dilation_level',default=5)
def run_graphcut_evaluation_for_weakly_supervised_learning(kernel_size,lamda,sigma,dilation_level):
    variable= "kernel_size,lamda,sigma,dilation_level"+str([kernel_size,lamda,sigma,dilation_level])
    variable=    variable.replace(' ','').replace(',','_').replace('[','_').replace(']','')
    print(variable)
    for i, (img, full_mask, weak_mask, _) in tqdm(enumerate(train_loader)):
        if weak_mask.sum() <= 0 or full_mask.sum() <= 0:
            continue
        img=img.cpu().data.squeeze().numpy()
        weak_mask=weak_mask.cpu().data.squeeze().numpy()
        full_mask = full_mask.cpu().data.squeeze().numpy()
        [db,df]=graphcut_with_FG_seed_and_BG_dlation(img,weak_mask,full_mask,kernal_size=kernel_size,lamda=lamda,sigma=sigma,dilation_level=dilation_level)
        db_meter.add(db)
        df_meter.add(df)
    db_average= db_meter.value()[0]
    df_average = df_meter.value()[0]
    print('db:%.3f, df:%.3f, mean dice:%.3f'%(db_average,df_average,0.5*(db_average+df_average)))
    data_to_save = pd.DataFrame([db_average, df_average, 0.5 * (db_average + df_average)]).T
    data_to_save.columns=['db','df','mean']
    data_to_save.to_csv('results/'+variable+'.csv')

if __name__=="__main__":
    run_graphcut_evaluation_for_weakly_supervised_learning()