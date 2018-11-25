# coding=utf-8
'''
this is to make sure that by showing their own results, the generalization ability can be improved.
'''
import copy
import os
import sys
import pandas as pd
import copy
sys.path.insert(-1, os.getcwd())
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import utils.medicalDataLoader as medicalDataLoader
from ADMM import ADMM_networks, ADMM_network_without_sizeConstraint, ADMM_network_without_graphcut
from utils.enet import Enet
from utils.network import UNet
from utils.criterion import CrossEntropyLoss2d
from utils.pretrain_network import pretrain
from utils.utils import Colorize, dice_loss, evaluate_iou, split_label_unlabel_dataset, iter_image_pair, \
    pred2segmentation

from tqdm import tqdm
import click

filename = os.path.basename(__file__).split('.')[0]

use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')

batch_size = 1
batch_size_val = 1
num_workers = 1
lr = 0.001
max_epoch = 100
data_dir = '/Users/jizong/workspace/DGA1032_grid_search/dataset/ACDC-2D-All'
inneriter = 10000

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

split_ratio = 0.05
labeled_dataset, unlabeled_dataset = split_label_unlabel_dataset(train_set, split_ratio)
labeled_dataLoader = DataLoader(labeled_dataset, batch_size=1, num_workers=num_workers, shuffle=True)
unlabeled_dataLoader = DataLoader(unlabeled_dataset, batch_size=1, num_workers=num_workers, shuffle=True)
unlabeled_dataLoader.dataset.augmentation = False

neural_net = Enet(2)
neural_net2 = copy.deepcopy(neural_net)

neural_net.to(device)
neural_net2.to(device)
ious_tables = []
criterion = nn.MSELoss()
criterion2 = CrossEntropyLoss2d(weight=torch.tensor([1, 1]).float()).to(device)
optimiser = torch.optim.Adam(neural_net.parameters(), lr=1e-3,weight_decay=1e-5)
optimiser2 = torch.optim.Adam(neural_net2.parameters(), lr=1e-3,weight_decay=1e-5)
for iteration in tqdm(range(10000)):

    (labeled_img, labeled_mask), (unlabeled_img, unlabeled_mask) = iter_image_pair(labeled_dataLoader,
                                                                                   unlabeled_dataLoader)

    # skip those with no foreground masks
    if labeled_mask.sum() <= 0 or unlabeled_mask.sum() <= 0:
        continue
    labeled_img, labeled_mask = labeled_img.to(device), labeled_mask.to(device)
    unlabeled_img, unlabeled_mask = unlabeled_img.to(device), unlabeled_mask.to(device)
    plt.ion()
    neural_net2.eval()

    loss_1 = []
    loss_2 = []

    for i in range(inneriter):
        neural_net.train()
        MSE_labeled_loss = criterion(F.softmax(neural_net(labeled_img), 1)[0, 1], labeled_mask.squeeze().float())
        MSE_labeled_loss2 = criterion(F.softmax(neural_net2(labeled_img), 1)[0, 1], labeled_mask.squeeze().float())

        CE_labeled_loss_1 = criterion2(neural_net(labeled_img),labeled_mask.squeeze(1))
        CE_labeled_loss_2 = criterion2(neural_net2(labeled_img),labeled_mask.squeeze(1))
        proba1 = F.softmax(neural_net(labeled_img), 1)[0, 1].cpu().data.numpy()

        plt.figure(1)
        plt.clf()
        plt.subplot(3, 2, 1)
        plt.imshow(labeled_img.cpu().squeeze().numpy())
        plt.subplot(3, 2, 2)
        plt.imshow(proba1)
        plt.colorbar()
        plt.subplot(3, 2, 3)
        plt.imshow(labeled_mask.cpu().data.squeeze().numpy())
        plt.subplot(3, 2, 4)
        plt.imshow(pred2segmentation(neural_net(labeled_img)).cpu().squeeze().numpy())
        plt.colorbar()

        plt.subplot(3,1,3)
        pd.Series(proba1.ravel()).plot.density()

        plt.show()
        plt.pause(0.01)


        proba2 = F.softmax(neural_net2(labeled_img), 1)[0, 1].cpu().data.numpy()
        plt.figure(2)
        plt.clf()
        plt.subplot(3, 2, 1)
        plt.imshow(labeled_img.cpu().squeeze().numpy())
        plt.subplot(3, 2, 2)
        plt.imshow(proba2)

        plt.colorbar()
        plt.subplot(3, 2, 3)
        plt.imshow(labeled_mask.cpu().data.squeeze().numpy())
        plt.subplot(3, 2, 4)
        plt.imshow(pred2segmentation(neural_net2(labeled_img)).squeeze().cpu().numpy())
        plt.colorbar()
        plt.subplot(3,1,3)
        pd.Series(proba2.ravel()).plot.density()

        plt.show()
        plt.pause(0.01)

        optimiser.zero_grad()
        optimiser2.zero_grad()
        # loss =  MSE_labeled_loss
        loss = CE_labeled_loss_1
        loss2 = CE_labeled_loss_2
        loss.backward()
        loss2.backward()
        optimiser.step()
        optimiser2.step()
        loss_1.append(loss.item())
        loss_2.append(loss2.item())

        if i % 10 == 0:
            plt.figure(3)
            plt.clf()
            plt.plot(loss_1, 'r', label='train')
            plt.plot(loss_2, 'b', label='eval')
            plt.legend()
            plt.show()

        # print(neural_net.encoder.)
