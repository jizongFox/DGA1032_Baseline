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
val_set = medicalDataLoader.MedicalImageDataset('val', data_dir, transform=transform, mask_transform=mask_transform,
                                                equalize=False)
val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)


@click.command()
@click.option('--baseline',default='ADMM', type=click.Choice(['ADMM', 'ADMM_size','ADMM_gc']))
@click.option('--inneriter', default=5, help='iterative time in an inner admm loop')
@click.option('--lamda', default=1.0, help='balance between unary and boundary terms')
@click.option('--sigma', default=0.02, help='sigma in the boundary term of the graphcut')
@click.option('--kernelsize', default=5, help='kernelsize of the graphcut')
@click.option('--lowbound', default=93, help='lowbound')
@click.option('--highbound', default=1728, help='highbound')
@click.option('--saved_name', default='default_iou', help='default_save_name')
def main(baseline, inneriter, lamda, sigma, kernelsize, lowbound, highbound, saved_name):
    ious_tables = []
    variable_str = str([baseline,inneriter, lamda, sigma, kernelsize, lowbound, highbound, saved_name]).replace(' ', '').replace(',', '_').replace("'", "").replace('[', '').replace(']', '')
    ious_tables.append([baseline,inneriter, lamda, sigma, kernelsize, lowbound, highbound, saved_name])

    # Here we have to split the fully annotated dataset and unannotated dataset
    split_ratio = 0.03
    random_index = np.random.permutation(len(train_set))
    labeled_dataset = copy.deepcopy(train_set)
    labeled_dataset.imgs = [train_set.imgs[x]
                            for x in random_index[:int(len(random_index) * split_ratio)]]
    unlabeled_dataset = copy.deepcopy(train_set)
    unlabeled_dataset.imgs = [train_set.imgs[x]
                              for x in random_index[int(len(random_index) * split_ratio):]]
    labeled_dataLoader = DataLoader(
        labeled_dataset, batch_size=1, num_workers=num_workers, shuffle=True)
    unlabeled_dataLoader = DataLoader(
        unlabeled_dataset, batch_size=1, num_workers=num_workers, shuffle=True)
    unlabeled_dataLoader.dataset.augmentation = False

    ##==================================================================================================================
    neural_net = Enet(2)

    map_location = lambda storage, loc: storage

    neural_net.load_state_dict(torch.load(
        'checkpoint/model_0.6649_split_0.030.pth', map_location=map_location))
    neural_net.to(device)

    plt.ion()
    if baseline =='ADMM':
        net = ADMM_networks(neural_net, lowerbound=lowbound, upperbound=highbound, lamda=lamda, sigma=sigma,
                            kernelsize=kernelsize)
    elif baseline=='ADMM_gc':
        net = ADMM_network_without_sizeConstraint(neural_net,lamda=lamda, sigma=sigma,kernelsize=kernelsize)

    elif baseline=='ADMM_size':
        net = ADMM_network_without_graphcut(neural_net,lowerbound=lowbound,upperbound=highbound)
    else:
        raise ValueError


    labeled_dataLoader_, unlabeled_dataLoader_ = iter(labeled_dataLoader), iter(unlabeled_dataLoader)
    for iteration in tqdm(range(10000)):
        # choose randomly a batch of image from labeled dataset and unlabeled dataset.
        # Initialize the ADMM dummy variables for one-batch training

        if (iteration ) % 200 == 0:
            unlabeled_ious = evaluate_iou(unlabeled_dataLoader, net.neural_net)
            val_ious = evaluate_iou(val_loader, net.neural_net)
            ious = np.array((unlabeled_ious, val_ious)).ravel().tolist()
            ious_tables.append(ious)
            try:
                if not os.path.exists(os.path.join('results',filename)):
                    os.mkdir(os.path.join('results',filename))

                pd.DataFrame(ious_tables).to_csv(os.path.join('results',filename,'%s.csv' % variable_str),header=None)
            except Exception as e:
                print(e)

        try:
            labeled_img, labeled_mask, labeled_weak_mask = next(labeled_dataLoader_)[0:3]
        except:
            labeled_dataLoader_ = iter(labeled_dataLoader)
            labeled_img, labeled_mask, labeled_weak_mask = next(labeled_dataLoader_)[0:3]

        labeled_img, labeled_mask, labeled_weak_mask = labeled_img.to(device), labeled_mask.to(
            device), labeled_weak_mask.to(device)

        try:
            unlabeled_img, unlabeled_mask = next(unlabeled_dataLoader_)[0:2]
        except:
            unlabeled_dataLoader_ = iter(unlabeled_dataLoader)
            unlabeled_img, unlabeled_mask = next(unlabeled_dataLoader_)[0:2]

        # skip those with no foreground masks
        if labeled_mask.sum() <= 0 or unlabeled_mask.sum() <= 0:
            continue
        unlabeled_img, unlabeled_mask = unlabeled_img.to(device), unlabeled_mask.to(device)

        for i in range(inneriter):
            net.neural_net.eval()
            net.update((labeled_img, labeled_mask),
                       (unlabeled_img, unlabeled_mask))
            # net.show_gamma()
        net.reset()


if __name__ == "__main__":
    np.random.seed(1)
    torch.random.manual_seed(1)
    main()
