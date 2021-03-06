import matplotlib.pyplot as plt
import maxflow
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pandas as pd

from utils.network import UNet
from utils.criterion import CrossEntropyLoss2d

use_gpu = True
device = torch.device('cuda') if torch.cuda.is_available() and use_gpu else torch.device('cpu')


class ADMM_networks(object):
    '''
    neural_net : neural network
    limage and uimage: b,c,h,w of torch.float
    gamma and s: numpy of shape b h w
    u,v: numpy of shape b h w
    '''
    def __init__(self, neural_network,lr, lowerbound, upperbound, lamda=1, sigma=0.02, kernelsize=5):
        super(ADMM_networks, self).__init__()
        self.lowbound = lowerbound
        self.upbound = upperbound
        self.neural_net = neural_network
        self.reset()
        self.optimiser = torch.optim.Adam(self.neural_net.parameters(), lr=lr)
        self.CEloss_criterion = CrossEntropyLoss2d()
        self.p_u = 10
        self.p_v = 10
        self.lamda = lamda
        self.sigma = sigma
        self.kernelsize = kernelsize
        self.initial_kernel()

    def learning_rate_decay(self,factor):
        assert factor>0 and factor<1
        lr = []
        for param_group in self.optimiser.param_groups:
            lr.append(param_group['lr'])
        new_lr=[x*factor for x in lr]
        # print(new_lr)
        for i, param_group in enumerate(self.optimiser.param_groups):
            param_group['lr'] = new_lr[i]



    def initial_kernel(self):
        self.kernel = np.ones((self.kernelsize, self.kernelsize))
        self.kernel[int(self.kernel.shape[0] / 2), int(self.kernel.shape[1] / 2)] = 0

    def limage_forward(self, limage, lmask):
        self.limage = limage
        self.lmask = lmask
        # self.neural_net.eval()
        # with torch.no_grad():
        self.limage_output = self.neural_net(limage)
        # self.neural_net.train()

    def uimage_forward(self, uimage, umask):
        self.umask = umask
        self.uimage = uimage
        # self.neural_net.eval()
        # with torch.no_grad():
        self.uimage_output = self.neural_net(uimage)
        # self.neural_net.train()

        if self.gamma is None:
            self.initialize_dummy_variables(self.uimage_output)

    def heatmap2segmentation(self, heatmap):
        return heatmap.max(1)[1]

    def initialize_dummy_variables(self, uimage_heatmap):
        self.gamma = self.heatmap2segmentation(uimage_heatmap).cpu().data.numpy()  # b, w, h
        self.s = self.gamma  # b w h
        self.u = np.zeros(list(self.gamma.shape))  # b w h
        self.v = np.zeros(self.u.shape)

    def reset(self):
        self.limage = None
        self.uimage = None
        self.lmask = None
        self.umask = None
        self.limage_output = None
        self.uimage_output = None
        self.gamma = None
        self.s = None
        self.u = None
        self.v = None
        # plt.close('all')

    def update_theta(self):

        self.neural_net.zero_grad()

        for i in range(3):
            CE_loss = self.CEloss_criterion(self.limage_output, self.lmask.squeeze(1))
            unlabled_loss = self.p_u / 2 * (F.softmax(self.uimage_output, dim=1)[:, 1] + torch.from_numpy(
                -self.gamma + self.u).float().to(device)).norm(p=2) ** 2 \
                            + self.p_v / 2 * (F.softmax(self.uimage_output, dim=1)[:, 1] + torch.from_numpy(
                -self.s + self.v).float().to(device)).norm(p=2) ** 2
            unlabled_loss /= list(self.uimage_output.reshape(-1).size())[0]

            loss = CE_loss+ unlabled_loss

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            self.uimage_forward(self.uimage, self.umask)
            self.limage_forward(self.limage, self.lmask)

    def set_boundary_term(self, g, nodeids, img, lumda, sigma):
        kernel = self.kernel
        transfer_function = lambda pixel_difference: lumda * np.exp((-1 / sigma ** 2) * pixel_difference ** 2)

        img = img.squeeze().cpu().data.numpy()

        # =====new =========================================
        padding_size = int(max(kernel.shape) / 2)
        position = np.array(list(zip(*np.where(kernel != 0))))

        def shift_matrix(matrix, kernel):
            center_x, center_y = int(kernel.shape[0] / 2), int(kernel.shape[1] / 2)
            [kernel_x, kernel_y] = np.array(list(zip(*np.where(kernel == 1))))[0]
            dy, dx = kernel_x - center_x, kernel_y - center_y
            shifted_matrix = np.roll(matrix, -dy, axis=0)
            shifted_matrix = np.roll(shifted_matrix, -dx, axis=1)
            return shifted_matrix

        for p in position[:int(len(position) / 2)]:
            structure = np.zeros(kernel.shape)
            structure[p[0], p[1]] = kernel[p[0], p[1]]
            pad_im = np.pad(img, ((padding_size, padding_size), (padding_size, padding_size)), 'constant',
                            constant_values=0)
            shifted_im = shift_matrix(pad_im, structure)
            weights_ = transfer_function(
                np.abs(pad_im - shifted_im)[padding_size:-padding_size, padding_size:-padding_size])

            g.add_grid_edges(nodeids, structure=structure, weights=weights_, symmetric=True)

        return g

    def update_gamma(self):

        unary_term_gamma_1 = np.multiply(
            (0.5 - (F.softmax(self.uimage_output, dim=1).cpu().data.numpy()[:, 1, :, :] + self.u)),
            1)

        unary_term_gamma_0 = np.zeros(unary_term_gamma_1.shape)
        new_gamma = np.zeros(self.gamma.shape)
        g = maxflow.Graph[float](0, 0)
        # Add the nodes.
        nodeids = g.add_grid_nodes(list(self.gamma.shape)[1:])
        # Add edges with the same capacities.

        # g.add_grid_edges(nodeids, neighbor_term)
        g = self.set_boundary_term(g, nodeids, self.uimage, lumda=self.lamda, sigma=self.sigma)

        # Add the terminal edges.
        g.add_grid_tedges(nodeids, (unary_term_gamma_0[0]).squeeze(),
                          (unary_term_gamma_1[0]).squeeze())
        g.maxflow()
        # Get the segments.
        sgm = g.get_grid_segments(nodeids) * 1

        # The labels should be 1 where sgm is False and 0 otherwise.
        new_gamma[0] = np.int_(np.logical_not(sgm))

        if new_gamma.sum() > 0:
            self.gamma = new_gamma
        else:
            self.gamma = self.s

    def update_s(self):
        a = 0.5 - (F.softmax(self.uimage_output, 1)[:, 1].cpu().data.numpy().squeeze() + self.v)
        original_shape = a.shape
        a_ = np.sort(a.ravel())
        useful_pixel_number = (a < 0).sum()
        if self.lowbound < useful_pixel_number and self.upbound > useful_pixel_number:
            self.s = ((a < 0) * 1.0).reshape(original_shape)
        if useful_pixel_number < self.lowbound:
            self.s = ((a <= a_[self.lowbound]) * 1).reshape(original_shape)
        if useful_pixel_number > self.upbound:
            self.s = ((a <= a_[self.upbound]) * 1).reshape(original_shape)

    def update_u(self):

        new_u = self.u + (F.softmax(self.uimage_output,dim=1)[0, 1].cpu().data.numpy() - self.gamma)*0.01
        self.u = new_u
        pass

    def update_v(self):
        new_v = self.v + (F.softmax(self.uimage_output,dim=1)[0, 1].cpu().data.numpy() - self.s)*0.01
        self.v = new_v
        pass

    def update(self, limage_pair, uimage_pair):
        [limage, lmask], [uimage, umask] = limage_pair, uimage_pair
        self.limage_forward(limage, lmask)
        self.uimage_forward(uimage, umask)
        self.update_s()
        self.update_gamma()
        self.update_theta()
        self.update_u()
        self.update_v()

    def update_1(self, limage_pair, uimage_pair):
        [limage, lmask], [uimage, umask] = limage_pair, uimage_pair
        self.limage_forward(limage, lmask)
        self.uimage_forward(uimage, umask)
        self.update_s()
        self.update_gamma()

    def update_2(self):
        self.update_theta()
        self.update_u()
        self.update_v()



    def show_labeled_pair(self):
        fig = plt.figure(1, figsize=(32, 32))
        plt.clf()
        fig.suptitle("labeled data", fontsize=16)

        ax1 = fig.add_subplot(221)
        ax1.imshow(self.limage[0].cpu().data.numpy().squeeze())
        ax1.title.set_text('original image')

        ax2 = fig.add_subplot(222)
        ax2.imshow(self.lmask[0].cpu().data.numpy().squeeze())
        ax2.title.set_text('ground truth')

        ax3 = fig.add_subplot(223)
        ax3.imshow(F.softmax(self.limage_output, dim=1)[0][1].cpu().data.numpy())
        ax3.title.set_text('prediction of the probability')

        ax4 = fig.add_subplot(224)
        ax4.imshow(np.abs(
            self.lmask[0].cpu().data.numpy().squeeze() - F.softmax(self.limage_output, dim=1)[0][1].cpu().data.numpy()))
        ax4.title.set_text('difference')

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    def show_ublabel_image(self):
        fig = plt.figure(2, figsize=(8, 8))
        plt.clf()
        fig.suptitle("Unlabeled data", fontsize=16)

        ax1 = fig.add_subplot(221)
        ax1.imshow(self.uimage[0].cpu().data.numpy().squeeze(), cmap='gray')
        ax1.title.set_text('original image')
        ax1.set_axis_off()

        ax2 = fig.add_subplot(222)
        ax2.cla()
        # ax1.imshow(self.uimage[0].cpu().data.numpy().squeeze(),cmap='gray')
        ax2.imshow(F.softmax(self.uimage_output, dim=1)[0][1].cpu().data.numpy(), vmin=0, vmax=1, cmap='gray')
        # ax2.contour(F.softmax(self.uimage_output, dim=1)[0][1].cpu().data.numpy(),level=(0.5,0.5),colors="red",alpha=0.5)
        ax2.title.set_text('probability prediction')
        ax2.text(0, 0, '%.3f' % F.softmax(self.uimage_output, dim=1)[0][1].cpu().data.numpy().max())
        # ax2.set_cl)
        ax2.set_axis_off()

        ax3 = fig.add_subplot(223)
        ax3.clear()
        ax3.imshow(self.uimage[0].cpu().data.numpy().squeeze(), cmap='gray')
        # ax3.imshow(self.umask.squeeze().cpu().data.numpy(),cmap='gray')
        ax3.contour(self.umask.squeeze().cpu().data.numpy(), level=[0], colors="red", alpha=1, linewidth=0.001)
        ax3.title.set_text('ground truth mask')
        ax3.set_axis_off()

        ax4 = fig.add_subplot(224)
        ax4.clear()
        ax4.imshow(self.uimage[0].cpu().data.numpy().squeeze(), cmap='gray')
        # ax4.imshow(self.heatmap2segmentation(self.uimage_output).squeeze().cpu().data.numpy(),cmap='gray')
        ax4.contour(self.heatmap2segmentation(self.uimage_output).squeeze().cpu().data.numpy(), level=[0.5],
                    colors="red", alpha=1, linewidth=0.001)
        # ax2.contour(F.softmax(self.uimage_output, dim=1)[0][1].cpu().data.numpy(),level=(0.5,0.5),colors="red",alpha=0.5)
        ax4.title.set_text('prediction mask')
        ax4.set_axis_off()
        # plt.tight_layout()
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()

        plt.show(block=False)
        plt.pause(0.01)

    def show_gamma(self):
        plt.figure(3, figsize=(5, 5))
        # plt.gray()
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.imshow(self.uimage[0].cpu().data.numpy().squeeze(), cmap='gray')
        # plt.imshow(self.gamma[0])
        plt.contour(self.umask.squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=1)

        plt.contour(self.s.squeeze(), level=[0], colors='blue', alpha=0.2, linewidth=0.001)

        plt.contour(self.gamma[0], level=[0], colors="red", alpha=0.2, linewidth=0.001)
        plt.contour(self.heatmap2segmentation(self.uimage_output).squeeze().cpu().data.numpy(), level=[0],
                    colors="green", alpha=0.2, linewidth=0.001)
        plt.title('Gamma')
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        plt.show(block=False)
        plt.pause(0.01)

    def show_u(self):
        plt.figure(4, figsize=(5, 5))
        plt.clf()
        plt.title('Multipicator')
        plt.subplot(1, 1, 1)
        plt.imshow(self.u.squeeze())
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.01)

    def show_s(self):
        plt.figure(5, figsize=(5, 5))
        plt.clf()
        plt.title('S size loss')
        plt.subplot(1, 1, 1)
        plt.imshow(self.s.squeeze())
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.01)

    def show_heatmap(self):
        plt.figure(10, figsize=(5, 5))
        # plt.gray()
        plt.clf()
        plt.subplot(1, 1, 1)
        # plt.imshow(self.image[0].cpu().data.numpy().squeeze(), cmap='gray')
        plt.imshow(F.softmax(self.uimage_output, dim=1)[:, 1].cpu().data.squeeze().numpy(), cmap='gray', alpha=0.5)
        plt.colorbar()
        # plt.contour(self.heatmap2segmentation(self.image_output).squeeze().cpu().data.numpy(), level=[0],
        #             colors="green", alpha=0.2, linewidth=0.001, label='CNN')
        plt.title('heatmap')
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        # plt.legend()
        plt.show(block=False)
        plt.pause(0.01)

class ADMM_network_without_sizeConstraint(ADMM_networks):

    def __init__(self, neural_network, lr, lamda=1, sigma=0.02, kernelsize=7):
        super().__init__(neural_network, lr=lr,lamda=lamda, sigma=sigma, kernelsize=kernelsize, lowerbound=0, upperbound=0, )

    def update_theta(self):
        self.neural_net.zero_grad()

        for i in range(5):
            CE_loss = self.CEloss_criterion(self.limage_output, self.lmask.squeeze(1))
            unlabled_loss = self.p_u / 2 * (F.softmax(self.uimage_output, dim=1)[:, 1] + torch.from_numpy(
                -self.gamma + self.u).float().to(device)).norm(p=2) ** 2
            unlabled_loss /= list(self.uimage_output.reshape(-1).size())[0]
            loss = CE_loss + unlabled_loss
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            self.uimage_forward(self.uimage, self.umask)
            self.limage_forward(self.limage, self.lmask)

    def update_s(self):
        pass

    def update_v(self):
        pass

    def update(self, limage_pair, uimage_pair):
        [limage, lmask], [uimage, umask] = limage_pair, uimage_pair
        self.limage_forward(limage, lmask)
        self.uimage_forward(uimage, umask)
        self.update_gamma()
        self.update_theta()
        self.update_u()

    def show_gamma(self):
        plt.figure(3, figsize=(5, 5))
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.imshow(self.uimage[0].cpu().data.numpy().squeeze(), cmap='gray')
        plt.contour(self.umask.squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=1)

        plt.contour(self.gamma[0], level=[0], colors="red", alpha=0.2, linewidth=0.001)
        plt.contour(self.heatmap2segmentation(self.uimage_output).squeeze().cpu().data.numpy(), level=[0],
                    colors="green", alpha=0.2, linewidth=0.001)
        plt.show(block=False)
        plt.pause(0.01)

class ADMM_network_without_graphcut(ADMM_networks):
    def __init__(self, neural_network, lr,lowerbound=50, upperbound=1723):
        super().__init__(neural_network, lr,lowerbound, upperbound, lamda=0, sigma=0, kernelsize=5)

    def update_theta(self):
        self.neural_net.zero_grad()

        for i in range(5):
            CE_loss = self.CEloss_criterion(self.limage_output, self.lmask.squeeze(1))
            unlabled_loss = self.p_v / 2 * (F.softmax(self.uimage_output, dim=1)[:, 1] + torch.from_numpy(
                -self.s + self.v).float().to(device)).norm(p=2) ** 2
            unlabled_loss /= list(self.uimage_output.reshape(-1).size())[0]
            loss = CE_loss + unlabled_loss
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            self.uimage_forward(self.uimage, self.umask)
            self.limage_forward(self.limage, self.lmask)

    def update_gamma(self):
        pass

    def update_u(self):
        pass

    def update(self, limage_pair, uimage_pair):
        [limage, lmask], [uimage, umask] = limage_pair, uimage_pair
        self.limage_forward(limage, lmask)
        self.uimage_forward(uimage, umask)
        self.update_s()
        self.update_theta()
        self.update_v()

    def show_gamma(self):
        plt.figure(3, figsize=(5, 5))
        # plt.gray()
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.imshow(self.uimage[0].cpu().data.numpy().squeeze(), cmap='gray')
        # plt.imshow(self.gamma[0])
        plt.contour(self.umask.squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=1)
        plt.contour(self.s.squeeze(), level=[0], colors='blue', alpha=0.2, linewidth=0.001)
        plt.contour(self.heatmap2segmentation(self.uimage_output).squeeze().cpu().data.numpy(), level=[0],
                    colors="green", alpha=0.2, linewidth=0.001)
        plt.show(block=False)
        plt.pause(0.01)

class weakly_ADMM_network(ADMM_networks):

    def __init__(self, neural_network, lr, lowerbound, upperbound, lamda=1, sigma=0.02, kernelsize=5,dilation_level = 7):
        super().__init__(neural_network, lr,lowerbound, upperbound, lamda, sigma, kernelsize)
        self.optimiser = torch.optim.Adam(self.neural_net.parameters(), lr=lr)
        self.CEloss_criterion = CrossEntropyLoss2d(torch.Tensor([0, 1]).float()).to(device)
        self.dilation_level = dilation_level

    def update(self, image_pair, full_mask):
        [image, weak_mask] = image_pair
        self.full_mask = full_mask
        self.image_forward(image, weak_mask)
        self.update_gamma()
        self.update_s()
        self.update_theta()
        self.update_u()
        self.update_v()

    def update_1(self, image_pair, full_mask):
        [image, weak_mask] = image_pair
        self.full_mask = full_mask
        self.image_forward(image, weak_mask)
        self.update_gamma()
        self.update_s()
    def update_2(self):
        self.update_theta()
        self.update_u()
        self.update_v()

    def update_gamma(self):
        unary_term_gamma_1 = np.multiply(
            (0.5 - (F.softmax(self.image_output, dim=1).cpu().data.numpy()[:, 1, :, :] + self.u)),
            1)
        unary_term_gamma_1[(self.weak_mask.squeeze(dim=1).cpu().data.numpy() == 1).astype(bool)] = -np.inf

        weak_mask = self.weak_mask.cpu().squeeze().numpy()

        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(weak_mask.astype(np.float32), kernel, iterations=self.dilation_level)
        unary_term_gamma_0 = np.zeros(unary_term_gamma_1.shape)
        unary_term_gamma_1[0][dilation != 1] = np.inf
        new_gamma = np.zeros(self.gamma.shape)
        g = maxflow.Graph[float](0, 0)
        nodeids = g.add_grid_nodes(list(self.gamma.shape)[1:])
        g = self.set_boundary_term(g, nodeids, self.image, lumda=self.lamda, sigma=self.sigma)
        g.add_grid_tedges(nodeids, (unary_term_gamma_0[0]).squeeze(),
                          (unary_term_gamma_1[0]).squeeze())
        g.maxflow()
        sgm = g.get_grid_segments(nodeids) * 1
        new_gamma[0] = np.int_(np.logical_not(sgm))
        if new_gamma.sum() > 0:
            self.gamma = new_gamma
        else:
            self.gamma = self.s

    def image_forward(self, image, weak_mask):
        self.weak_mask = weak_mask
        self.image = image
        self.image_output = self.neural_net(image)
        if self.gamma is None:
            self.initialize_dummy_variables(self.image_output)

    def reset(self):
        self.image = None
        self.weak_mask = None
        self.image_output = None
        self.gamma = None
        self.s = None
        self.u = None
        self.v = None

    def update_s(self):
        a = 0.5 - (F.softmax(self.image_output, 1)[:, 1].cpu().data.numpy().squeeze() + self.v)
        original_shape = a.shape
        a_ = np.sort(a.ravel())
        useful_pixel_number = (a < 0).sum()
        if self.lowbound < useful_pixel_number and self.upbound > useful_pixel_number:
            self.s = ((a < 0) * 1.0).reshape(original_shape)
        if useful_pixel_number < self.lowbound:
            self.s = ((a <= a_[self.lowbound]) * 1).reshape(original_shape)
        if useful_pixel_number > self.upbound:
            self.s = ((a <= a_[self.upbound]) * 1).reshape(original_shape)

    def update_theta(self):

        self.neural_net.zero_grad()

        for i in range(5):
            CE_loss = self.CEloss_criterion(self.image_output, self.weak_mask.squeeze(1).long())
            unlabled_loss = self.p_u / 2 * (F.softmax(self.image_output, dim=1)[:, 1] + torch.from_numpy(
                -self.gamma + self.u).float().to(device)).norm(p=2) ** 2 \
                            + self.p_v / 2 * (F.softmax(self.image_output, dim=1)[:, 1] + torch.from_numpy(
                -self.s + self.v).float().to(
                device)).norm(p=2) ** 2

            unlabled_loss /= list(self.image_output.reshape(-1).size())[0]

            loss = CE_loss + unlabled_loss
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            self.image_forward(self.image, self.weak_mask)

    def update_u(self):
        new_u = self.u + (F.softmax(self.image_output, dim=1)[:, 1, :, :].cpu().data.numpy() - self.gamma)*0.01
        self.u = new_u

    def update_v(self):
        new_v = self.v + (F.softmax(self.image_output, dim=1)[:, 1, :, :].cpu().data.numpy() - self.s)*0.01
        self.v = new_v

    def show_gamma(self):
        plt.figure(3, figsize=(5, 5))
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.imshow(self.image[0].cpu().data.numpy().squeeze(), cmap='gray')

        plt.contour(self.weak_mask.squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')
        plt.contour(self.full_mask.squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')

        plt.contour(self.gamma[0], level=[0], colors="red", alpha=0.2, linewidth=0.001, label='graphcut')
        plt.contour(self.s.squeeze(), level=[0], colors='blue', alpha=0.2, linewidth=0.001, label='size_constraint')
        plt.contour(self.heatmap2segmentation(self.image_output).squeeze().cpu().data.numpy(), level=[0],
                    colors="green", alpha=0.2, linewidth=0.001, label='CNN')
        plt.title('Gamma')
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        # plt.legend()
        plt.show(block=False)
        plt.pause(0.01)

    def show_heatmap(self):
        plt.figure(10, figsize=(5, 5))
        # plt.gray()
        plt.clf()
        plt.subplot(1, 1, 1)
        # plt.imshow(self.image[0].cpu().data.numpy().squeeze(), cmap='gray')
        plt.imshow(F.softmax(self.image_output, dim=1)[:, 1].cpu().data.squeeze().numpy(), cmap='gray', alpha=0.5)
        plt.colorbar()
        # plt.contour(self.heatmap2segmentation(self.image_output).squeeze().cpu().data.numpy(), level=[0],
        #             colors="green", alpha=0.2, linewidth=0.001, label='CNN')
        plt.title('heatmap')
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        # plt.legend()
        plt.show(block=False)
        plt.pause(0.01)

class weakly_ADMM_without_sizeConstraint(weakly_ADMM_network):

    def __init__(self, neural_network,lr, lamda=1.0, sigma=0.02, kernelsize=5,dilation_level=7):
        super().__init__(neural_network, lr,lowerbound=0, upperbound=0, lamda=lamda, sigma=sigma, kernelsize=kernelsize,dilation_level=dilation_level)

    def update_theta(self):
        self.neural_net.zero_grad()
        for i in range(5):
            CE_loss = self.CEloss_criterion(self.image_output, self.weak_mask.squeeze(1).long())
            unlabled_loss = self.p_u / 2 * (F.softmax(self.image_output, dim=1)[:, 1] + torch.from_numpy(
                -self.gamma + self.u).float().to(device)).norm(p=2) ** 2

            unlabled_loss /= list(self.image_output.reshape(-1).size())[0]

            loss = CE_loss + unlabled_loss
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            self.image_forward(self.image, self.weak_mask)

    def update_v(self):
        pass

    def update_s(self):
        pass

    def update(self, image_pair, full_mask):
        [image, weak_mask] = image_pair
        self.full_mask = full_mask
        self.image_forward(image, weak_mask)
        self.update_gamma()
        self.update_theta()
        self.update_u()

    def update_1(self, image_pair, full_mask):
        [image, weak_mask] = image_pair
        self.full_mask = full_mask
        self.image_forward(image, weak_mask)
        self.update_gamma()

    def update_2(self):
        self.update_theta()
        self.update_u()

    def show_gamma(self):
        plt.figure(3, figsize=(5, 5))
        # plt.gray()
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.imshow(self.image[0].cpu().data.numpy().squeeze(), cmap='gray')
        # plt.imshow(self.gamma[0])
        plt.contour(self.weak_mask.squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')
        plt.contour(self.full_mask.squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')

        plt.contour(self.gamma[0], level=[0], colors="red", alpha=0.2, linewidth=0.001, label='graphcut')

        plt.contour(self.heatmap2segmentation(self.image_output).squeeze().cpu().data.numpy(), level=[0],
                    colors="green", alpha=0.2, linewidth=0.001, label='CNN')
        plt.title('Gamma')
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        # plt.legend()
        plt.show(block=False)
        plt.pause(0.01)

class weakly_ADMM_without_gc(weakly_ADMM_network):

    def __init__(self,neural_network,lr ,lowerbound, upperbound):
        super().__init__(neural_network,lr, lowerbound, upperbound, lamda=1, sigma=1, kernelsize=5)

    def update(self, image_pair, full_mask):
        [image, weak_mask] = image_pair
        self.full_mask = full_mask
        self.image_forward(image, weak_mask)
        self.update_s()
        self.update_theta()
        self.update_v()

    def update_theta(self):
        self.neural_net.zero_grad()

        for i in range(5):
            CE_loss = self.CEloss_criterion(self.image_output, self.weak_mask.squeeze(1).long())
            unlabled_loss = self.p_v / 2 * (
                        F.softmax(self.image_output, dim=1)[:, 1] + torch.from_numpy(-self.s + self.v).float().to(
                    device)).norm(p=2) ** 2

            unlabled_loss /= list(self.image_output.reshape(-1).size())[0]

            loss = CE_loss + unlabled_loss
            # loss = unlabled_loss
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            # print(loss.item())

            self.image_forward(self.image, self.weak_mask)

    def update_gamma(self):
        pass

    def update_u(self):
        pass

    def show_gamma(self):
        plt.figure(3, figsize=(5, 5))
        # plt.gray()
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.imshow(self.image[0].cpu().data.numpy().squeeze(), cmap='gray')
        # plt.imshow(self.gamma[0])
        plt.contour(self.weak_mask.squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')
        plt.contour(self.full_mask.squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')

        plt.contour(self.s.squeeze(), level=[0], colors='blue', alpha=0.2, linewidth=0.001, label='size_constraint')
        plt.contour(self.heatmap2segmentation(self.image_output).squeeze().cpu().data.numpy(), level=[0],
                    colors="green", alpha=0.2, linewidth=0.001, label='CNN')
        plt.title('Gamma')

        plt.show(block=False)
        plt.pause(0.01)