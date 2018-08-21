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
from ADMM import ADMM_networks, ADMM_network_without_sizeConstraint, ADMM_network_without_graphcut,weakly_ADMM_network
from utils.enet import Enet
from utils.pretrain_network import pretrain
from utils.utils import Colorize, dice_loss, evaluate_iou

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
train_loader = DataLoader(train_set,batch_size= batch_size, shuffle= True, num_workers= num_workers)
val_set = medicalDataLoader.MedicalImageDataset('val', data_dir, transform=transform, mask_transform=mask_transform,
                                                equalize=False)
val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)


@click.command()
@click.option('--baseline',default='ADMM', type=click.Choice(['ADMM', 'ADMM_size','ADMM_gc']))
@click.option('--inneriter', default=5, help='iterative time in an inner admm loop')
@click.option('--lamda', default=1.0, help='balance between unary and boundary terms')
@click.option('--sigma', default=0.2, help='sigma in the boundary term of the graphcut')
@click.option('--kernelsize', default=5, help='kernelsize of the graphcut')
@click.option('--lowbound', default=93, help='lowbound')
@click.option('--highbound', default=1728, help='highbound')
@click.option('--saved_name', default='default_iou', help='default_save_name')
def main(baseline, inneriter, lamda, sigma, kernelsize, lowbound, highbound, saved_name):
    ious_tables = []
    variable_str = str([baseline,inneriter, lamda, sigma, kernelsize, lowbound, highbound, saved_name]).replace(' ', '').replace(',', '_').replace("'", "").replace('[', '').replace(']', '')
    ious_tables.append([baseline,inneriter, lamda, sigma, kernelsize, lowbound, highbound, saved_name])

    ##==================================================================================================================
    neural_net = Enet(2)

    map_location = lambda storage, loc: storage

    # neural_net.load_state_dict(torch.load(
    #     'checkpoint/model_0.6649_split_0.030.pth', map_location=map_location))
    neural_net.to(device)
    net = weakly_ADMM_network(neural_net, lowerbound=50, upperbound=2000,sigma=sigma,lamda=lamda)
    plt.ion()

    for i, (img, full_mask, weak_mask, _) in tqdm(enumerate(train_loader)):
        if weak_mask.sum() <= 0 or full_mask.sum() <= 0:
            continue
        img, full_mask, weak_mask = img.to(device), full_mask.to(device), weak_mask.to(device)


        for i in range(inneriter):
            net.neural_net.eval()
            net.update((img, weak_mask), full_mask)
            net.show_gamma()
            # net.show_heatmap()
            pass
        net.reset()


if __name__ == "__main__":
    np.random.seed(1)
    torch.random.manual_seed(1)
    main()
