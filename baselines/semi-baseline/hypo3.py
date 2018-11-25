# coding=utf-8
import copy, os, sys, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import time
sys.path.insert(-1, os.path.basename(os.path.dirname(os.getcwd())))
# import utils.medicalDataLoader as medicalDataLoader
import sklearn
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

torch.set_num_threads(1)
from sklearn.datasets import load_iris

data = load_iris()

X = data['data'][:, 0:2]
y = data['target']


class Iris_dataset(Dataset):

    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


dataset = Iris_dataset(X=X, y=y)
data_Loader = DataLoader(dataset=dataset, batch_size=150, shuffle=True, num_workers=4)


def show_decision_boundary(net,fignum=1):
    net.eval()
    # Plotting decision regions
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    fig = plt.figure(fignum, figsize=(10, 8))
    axs = fig.subplots(1, 1, sharex='col', sharey='row', )

    Z = F.softmax(net(torch.from_numpy(np.stack((xx.ravel(), yy.ravel()), axis=0).T).float()), 1)
    results = Z.max(1)[1]
    results = results.data.numpy().reshape(xx.shape)
    original_shape = list(xx.shape)
    original_shape.append(3)
    Z = Z.data.numpy().reshape(original_shape)

    axs.contourf(xx, yy, results, alpha=0.4)
    axs.scatter(X[:, 0], X[:, 1], c=data_Loader.dataset.y,
                s=20, edgecolor='k')

    plt.show()
    plt.pause(0.001)
    net.train()


class network(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 50),
            nn.BatchNorm1d(50),
            nn.PReLU(),
            # nn.ReLU(),
            nn.Linear(50, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),

            nn.Linear(50, 3)
        )

    def forward(self, input):
        output = self.encoder(input)
        return output


plt.ion()
net1 = network()
net2 = copy.deepcopy(net1)
net2.eval()
criterion = nn.CrossEntropyLoss()
optimiser1 = torch.optim.Adam(net1.parameters(), lr=5e-3)
optimiser2 = torch.optim.Adam(net2.parameters(), lr=5e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser1,milestones=[1000,2000],gamma=0.5)
time_begin = time.time()
for j in range(10000):
    scheduler.step()
    for i, (batch_x, batch_y) in enumerate(data_Loader):
        optimiser1.zero_grad()
        # net1.eval()
        output1 = net1(batch_x.float())
        # net1.train()
        loss1 = criterion(output1, batch_y.long())
        loss1.backward()
        optimiser1.step()
        # optimiser2.zero_grad()
        # output2 = net2(batch_x.float())
        # loss2 = criterion(output2, batch_y)
        # loss2.backward()
        # optimiser2.step()
        print('epoch:%d, loss:%.4f'%(j,loss1.item()))
    if j==5000:
        print('cost time:', time.time()-time_begin)
        break


    # if j ==10:
    #     net2.eval()
    #     print('eval')
    #     data_Loader.dataset.y=np.random.randint(0,3,data_Loader.dataset.y.shape)
    #     criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,3,1]).float())

    #
    # if j %10==0:
    #     show_decision_boundary(net1,1)
    #     show_decision_boundary(net2,2)
