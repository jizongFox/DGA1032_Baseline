import os
import sys
from torchnet.meter import AverageValueMeter
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
import warnings
from utils.enet import Enet

warnings.filterwarnings('ignore')
import torch
import copy
from torch.utils.data import DataLoader
from torchvision import transforms
import utils.medicalDataLoader as medicalDataLoader
from utils.utils import Colorize, evaluate_iou, split_label_unlabel_dataset, iter_image_pair
from utils.pretrain_network import pretrain
from tqdm import tqdm
import click

filename = os.path.basename(__file__).split('.')[0]
data_dir = '/Users/jizong/workspace/DGA1032_grid_search/dataset/ACDC-2D-All'
use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')

batch_size = 1
batch_size_val = 1
num_workers = 1
max_image_pairs = 10000

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

neural_net = Enet(2)

db_meter = AverageValueMeter()
df_meter = AverageValueMeter()

## pretrain the network on the labeled data
@click.command()
@click.option('--sr', default=0.03)
def run_pretrain(sr):
    split_ratio = sr
    labeled_dataset, unlabeled_dataset = split_label_unlabel_dataset(train_set, split_ratio)
    labeled_dataLoader = DataLoader(labeled_dataset, batch_size=1, num_workers=num_workers, shuffle=True)
    # unlabeled_dataLoader = DataLoader(unlabeled_dataset, batch_size=1, num_workers=num_workers, shuffle=True)
    pretrain(labeled_dataLoader, val_loader, neural_net, split_ratio=split_ratio, path='')




if __name__ == "__main__":
    run_pretrain()