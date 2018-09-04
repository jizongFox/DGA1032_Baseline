# coding=utf8
import os
import sys
from torchvision.utils import save_image,make_grid
import pandas as pd
from tensorboardX import SummaryWriter
sys.path.insert(-1, os.getcwd())
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import utils.medicalDataLoader as medicalDataLoader
from ADMM import weakly_ADMM_network, weakly_ADMM_without_sizeConstraint, weakly_ADMM_without_gc
from utils.enet import Enet
from utils.utils import Colorize, evaluate_iou

from tqdm import tqdm
import click
torch.set_num_threads(1)

filename = os.path.basename(__file__).split('.')[0]
writer = SummaryWriter('log/weakly')
use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')

batch_size = 1
batch_size_val = 1
num_workers = 1
lr = 0.001
max_epoch = 300
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
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_set = medicalDataLoader.MedicalImageDataset('val', data_dir, transform=transform, mask_transform=mask_transform,
                                                equalize=False)
val_loader = DataLoader(val_set, batch_size=batch_size_val, num_workers=num_workers, shuffle=True)


@click.command()
@click.option('--baseline', default='ADMM_weak', type=click.Choice(['ADMM_weak', 'ADMM_weak_gc', 'ADMM_weak_size']))
@click.option('--inneriter', default=1, help='iterative time in an inner admm loop')
@click.option('--lamda', default=1.0, help='balance between unary and boundary terms')
@click.option('--sigma', default=0.01, help='sigma in the boundary term of the graphcut')
@click.option('--kernelsize', default=7, help='kernelsize of the graphcut')
@click.option('--dilation_level', default=7, help='dilation_level of the graphcut')
@click.option('--lowbound', default=93, help='lowbound')
@click.option('--highbound', default=1728, help='highbound')
@click.option('--assign_size_to_each', default=True, help='default_save_name')
@click.option('--eps',default=0.05,help='default eps for testing')
def main(baseline, inneriter, lamda, sigma, kernelsize, dilation_level, lowbound, highbound, assign_size_to_each,eps):
    ious_tables = []
    variable_str = str([baseline, inneriter, lamda, sigma, kernelsize, dilation_level, lowbound, highbound, assign_size_to_each,eps]).replace(' ',
                                                                                                                 '').replace(
        ',', '_').replace("'", "").replace('[', '').replace(']', '')
    ious_tables.append([baseline, inneriter, lamda, sigma, kernelsize, dilation_level, lowbound, highbound, assign_size_to_each,eps])

    ##==================================================================================================================
    neural_net = Enet(2)
    neural_net.to(device)

    if baseline == 'ADMM_weak':
        net = weakly_ADMM_network(neural_net, lr, lowerbound=lowbound, upperbound=highbound, sigma=sigma, lamda=lamda,dilation_level=dilation_level,assign_size_to_each=assign_size_to_each,eps=eps)
    elif baseline == 'ADMM_weak_gc':
        net = weakly_ADMM_without_sizeConstraint(neural_net, lr, lamda=lamda, sigma=sigma, kernelsize=kernelsize,dilation_level=dilation_level)
    elif baseline == 'ADMM_weak_size':
        net = weakly_ADMM_without_gc(neural_net, lr, lowerbound=lowbound, upperbound=highbound,assign_size_to_each=assign_size_to_each,eps=eps)
    else:
        raise ValueError

    plt.ion()
    for iteration in range(max_epoch):

        [train_ious,train_grid] = evaluate_iou(train_loader, net.neural_net,save=True)
        [val_ious,val_grid] = evaluate_iou(val_loader, net.neural_net,save=True)
        writer.add_scalar('data/train_f_dice', train_ious[1], iteration)
        writer.add_scalar('data/val_f_dice', val_ious[1], iteration)

        try:
            save_image(train_grid,os.path.join('results',filename,'%strain_grid_%.2d_f_dice_%.3f.png'%(variable_str,iteration,train_ious[1])))
            save_image(val_grid,os.path.join('results',filename,'%sval_grid_%.2d_f_dice_%.3f.png'%(variable_str,iteration,val_ious[1])))
        except Exception as e:
            print(e)

        ious = np.array((train_ious, val_ious)).ravel().tolist()
        ious_tables.append(ious)
        try:
            if not os.path.exists(os.path.join('results', filename)):
                os.mkdir(os.path.join('results', filename))

            pd.DataFrame(ious_tables).to_csv(os.path.join('results', filename, '%s.csv' % variable_str), header=None)
        except Exception as e:
            print(e)

        if iteration%20 ==0:
            net.learning_rate_decay(0.9)

        for j, (img, full_mask, weak_mask, _) in tqdm(enumerate(train_loader)):
            if weak_mask.sum() <= 0 or full_mask.sum() <= 0:
                continue
            img, full_mask, weak_mask = img.to(device), full_mask.to(device), weak_mask.to(device)

            for i in range(inneriter):
                net.update_1((img, weak_mask), full_mask)
                net.show_gamma()
                net.show_heatmap()
                # print(net.upbound,net.lowbound)
                net.update_2()
            net.reset()



if __name__ == "__main__":
    np.random.seed(1)
    torch.random.manual_seed(1)
    main()
