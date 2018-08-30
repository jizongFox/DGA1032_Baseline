# coding=utf8
import copy
import os
import sys
import pandas as pd

sys.path.insert(-1, os.getcwd())
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import utils.medicalDataLoader as medicalDataLoader
from ADMM import ADMM_networks, ADMM_network_without_sizeConstraint, ADMM_network_without_graphcut
from utils.enet import Enet
from utils.pretrain_network import pretrain
from utils.utils import Colorize, dice_loss, evaluate_iou,split_label_unlabel_dataset,iter_image_pair,graphcut_refinement

from tqdm import tqdm
import click

filename = os.path.basename(__file__).split('.')[0]

use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')

batch_size = 1
batch_size_val = 1
num_workers = 1
lr = 0.01
max_epoch = 100
data_dir = 'dataset/ACDC-2D-All'



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
val_set = medicalDataLoader.MedicalImageDataset('val', data_dir, transform=transform, mask_transform=mask_transform,
                                                equalize=False)
val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)

@click.command()
@click.option('--split_ratio',default='0.05')
def main(split_ratio):
    # Here we have to split the fully annotated dataset and unannotated dataset
    labeled_dataset, unlabeled_dataset=split_label_unlabel_dataset(train_set,float(split_ratio))
    labeled_dataLoader = DataLoader(labeled_dataset, batch_size=1, num_workers=num_workers, shuffle=True)
    unlabeled_dataLoader = DataLoader(unlabeled_dataset, batch_size=1, num_workers=num_workers, shuffle=True)
    unlabeled_dataLoader.dataset.augmentation = False
    ##==================================================================================================================
    neural_net = Enet(2)
    neural_net.to(device)
    pretrain(labeled_dataLoader,val_loader,neural_net,lr=5e-4, split_ratio=split_ratio)

if __name__ == "__main__":
    main()
