import os
import sys,pandas as pd
from torchnet.meter import AverageValueMeter
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
import warnings
from utils.enet import Enet

warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader
from torchvision import transforms
import utils.medicalDataLoader as medicalDataLoader
from utils.utils import Colorize, evaluate_iou, split_label_unlabel_dataset, iter_image_pair,graphcut_refinement,dice_loss,pred2segmentation
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

db_meter_g = AverageValueMeter()
df_meter_g = AverageValueMeter()
db_meter_n = AverageValueMeter()
df_meter_n = AverageValueMeter()
## pretrain the network on the labeled data
@click.command()
@click.option('--model_name', default='model_0.6217_split_0.030.pth')
@click.option('--lamda', default=10.0, help='balance between unary and boundary terms')
@click.option('--sigma', default=0.01, help='sigma in the boundary term of the graphcut')
@click.option('--kernelsize', default=5, help='kernelsize of the graphcut')
def run_pretrain(model_name,lamda,sigma,kernelsize):
    variable_str = str([lamda,sigma,kernelsize]).replace(' ', '').replace(',', '_').replace("'", "").replace('[', '').replace(']', '')
    neural_net = Enet(2)
    map_location = lambda storage, loc: storage
    model_path = '/Users/jizong/workspace/DGA1032_grid_search/semi_pretrain_checkpoint/'+model_name
    father_path = os.path.dirname(model_path)
    neural_net.load_state_dict(torch.load(model_path, map_location=map_location))
    neural_net.to(device)
    # neural_net.eval()
    for i,(img,mask,_,_) in tqdm(enumerate(val_loader)):
        # if mask.sum()==0:
        #     continue

        proba = F.softmax(neural_net(img),1)[0,1]

        graphcut_output = graphcut_refinement(proba,img,5,10,0.01)
        [db,df] = dice_loss(graphcut_output,mask)
        db_meter_g.add(db)
        df_meter_g.add(df)
        seg = (proba>0.5)*1
        seg=seg.reshape(graphcut_output.shape).to(torch.int64)
        [db,df] = dice_loss(seg,mask)
        db_meter_n.add(db)
        df_meter_n.add(df)
    if not os.path.exists(father_path+'/semi_baseline_results'):
        try:
            os.mkdir(father_path+'/semi_baseline_results')
        except Exception as e:
            print(e)

    table= pd.DataFrame([db_meter_n.value()[0],df_meter_n.value()[0],db_meter_g.value()[0],df_meter_g.value()[0]]).T
    table.columns=['neural_b_dice','neural_f_dice','graphcut_b_dice','graphcut_f_dice']
    table.to_csv(father_path+'/semi_baseline_results/'+model_name.replace('pth','')+variable_str+'.csv')


        # plt.figure(3, figsize=(5, 5))
        # # plt.gray()
        # plt.clf()
        # plt.subplot(1, 1, 1)
        # plt.imshow(img[0].cpu().data.numpy().squeeze(), cmap='gray')
        # # plt.imshow(self.gamma[0])
        # plt.contour(mask.squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=1)
        #
        # plt.contour(graphcut_output.squeeze().cpu().data.numpy(), level=[0], colors="red", alpha=0.2, linewidth=0.001)
        # plt.contour(seg.squeeze().cpu().data.numpy(), level=[0],
        #             colors="green", alpha=0.2, linewidth=0.001)
        # plt.title('Gamma')
        # # figManager = plt.get_current_fig_manager()
        # # figManager.window.showMaximized()
        # plt.show(block=False)
        # plt.pause(0.01)


        # if df<0.8:
        #     print()

    print('graphcut_output: fiou:%.2f, direct_output: fiou:%.2f'%(df_meter_g.value()[0],df_meter_n.value()[0]))







if __name__ == "__main__":
    run_pretrain()