import torch, os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchnet.meter import AverageValueMeter

from utils.criterion import  CrossEntropyLoss2d
from utils.utils import dice_loss

use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')


def val(val_dataloader, network):

    network.eval()
    dice_meter = AverageValueMeter()
    dice_meter.reset()
    for i, (image, mask,_,_) in enumerate(val_dataloader):
        image,mask = image.to(device),mask.to(device)
        proba = F.softmax(network(image),dim=1)
        predicted_mask = proba.max(1)[1]
        iou = dice_loss(predicted_mask, mask).item()  ## this is the foreground iou loss
        dice_meter.add(iou)
    network.train()
    print('val iou:  %.8f'%dice_meter.value()[0])
    return dice_meter.value()[0]


def pretrain(train_dataloader, val_dataloader_, network, path=None, split_ratio=0.1):
    highest_iou = -1
    class config:
        lr = 1e-3
        epochs = 100
        path = 'checkpoint'


    pretrain_config = config()
    if path :
        pretrain_config.path = path
    network.to(device)
    criterion_ = CrossEntropyLoss2d()
    optimiser_ = torch.optim.Adam(network.parameters(),pretrain_config.lr)
    loss_meter = AverageValueMeter()
    fiou_tables = []

    for iteration in range(pretrain_config.epochs):
        loss_meter.reset()

        for i, (img,mask,weak_mask,_) in tqdm(enumerate(train_dataloader)):
            img,mask = img.to(device), mask.to(device)
            optimiser_.zero_grad()
            output = network(img)
            loss = criterion_(output,mask.squeeze(1))
            loss.backward()
            optimiser_.step()
            loss_meter.add(loss.item())
        print('train_loss: %.6f'%loss_meter.value()[0])

        if (iteration+1) %50 ==0:
            for param_group in optimiser_.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
                print('learning rate:', param_group['lr'])

        val_iou = val(val_dataloader_,network)
        fiou_tables.append(val_iou)
        if val_iou > highest_iou:
            highest_iou = val_iou
            torch.save(network.state_dict(),
                       os.path.join(pretrain_config.path, 'model_%.4f_split_%.3f.pth' % (val_iou, split_ratio)))
            print('pretrained model saved with %.4f.'%highest_iou)
    return fiou_tables
