import matplotlib.pyplot as plt
import maxflow
import numpy as np
import torch
import torch.nn.functional as F

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

    def __init__(self, neural_network, lowerbound, upperbound, lamda=1, sigma=0.02, kernelsize=5):
        super(ADMM_networks, self).__init__()
        self.lowbound = lowerbound
        self.upbound = upperbound
        self.neural_net = neural_network
        self.reset()
        self.optimiser = torch.optim.Adam(self.neural_net.parameters(), lr=0.0001)
        self.CEloss_criterion = CrossEntropyLoss2d()
        self.p_u = 1.0
        self.p_v = 1.0
        self.lamda = lamda
        self.sigma = sigma
        self.kernelsize = kernelsize
        self.initial_kernel()

    def initial_kernel(self):
        self.kernel = np.ones((self.kernelsize, self.kernelsize))
        self.kernel[int(self.kernel.shape[0] / 2), int(self.kernel.shape[1] / 2)] = 0

    def limage_forward(self, limage, lmask):
        self.limage = limage
        self.lmask = lmask
        self.limage_output = self.neural_net(limage)

    def uimage_forward(self, uimage, umask):
        self.umask = umask
        self.uimage = uimage
        self.uimage_output = self.neural_net(uimage)

        if self.gamma is None:
            self.__initialize_dummy_variables(self.uimage_output)

    def heatmap2segmentation(self, heatmap):
        return heatmap.max(1)[1]

    def __initialize_dummy_variables(self, uimage_heatmap):
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

        for i in range(5):
            CE_loss = self.CEloss_criterion(self.limage_output, self.lmask.squeeze(1))
            unlabled_loss = self.p_u / 2 * (F.softmax(self.uimage_output, dim=1)[:, 1] + torch.from_numpy(
                -self.gamma + self.u).float().to(device)).norm(p=2) ** 2 \
                            + self.p_v / 2 * (F.softmax(self.uimage_output, dim=1)[:, 1] + torch.from_numpy(
                -self.s + self.v).float().to(device)).norm(p=2) ** 2
            unlabled_loss /= list(self.uimage_output.reshape(-1).size())[0]

            loss = CE_loss + unlabled_loss
            # loss = unlabled_loss
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            # print('loss:', loss.item())

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

        for p in position:
            structure = np.zeros(kernel.shape)
            structure[p[0], p[1]] = kernel[p[0], p[1]]
            pad_im = np.pad(img, ((padding_size, padding_size), (padding_size, padding_size)), 'constant',
                            constant_values=0)
            shifted_im = shift_matrix(pad_im, structure)
            weights_ = transfer_function(
                np.abs(pad_im - shifted_im)[padding_size:-padding_size, padding_size:-padding_size])

            g.add_grid_edges(nodeids, structure=structure, weights=weights_, symmetric=False)

        return g

    def update_gamma(self):

        unary_term_gamma_1 = np.multiply(
            (0.5 - (F.softmax(self.uimage_output, dim=1).cpu().data.numpy()[:, 1, :, :] + self.u)),
            1)

        unary_term_gamma_0 = np.zeros(unary_term_gamma_1.shape)
        new_gamma = np.zeros(self.gamma.shape)
        g = maxflow.Graph[float](0, 0)
        i = 0
        # Add the nodes.
        nodeids = g.add_grid_nodes(list(self.gamma.shape)[1:])
        # Add edges with the same capacities.

        # g.add_grid_edges(nodeids, neighbor_term)
        g = self.set_boundary_term(g, nodeids, self.uimage, lumda=self.lamda, sigma=self.sigma)

        # Add the terminal edges.
        g.add_grid_tedges(nodeids, (unary_term_gamma_0[i]).squeeze(),
                          (unary_term_gamma_1[i]).squeeze())
        g.maxflow()
        # Get the segments.
        sgm = g.get_grid_segments(nodeids) * 1

        # The labels should be 1 where sgm is False and 0 otherwise.
        new_gamma[i] = np.int_(np.logical_not(sgm))
        # g.reset()
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

        new_u = self.u + (self.uimage_output[0, 1].cpu().data.numpy() - self.gamma) * 0.01
        self.u = new_u
        pass

    def update_v(self):
        new_v = self.v + (self.uimage_output[0, 1].cpu().data.numpy() - self.s) * 0.001
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
        plt.imshow(np.abs(self.u.squeeze()))
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.01)

    def show_s(self):
        plt.figure(5, figsize=(5, 5))
        plt.clf()
        plt.title('Multipicator')
        plt.subplot(1, 1, 1)
        plt.imshow(np.abs(self.s.squeeze()))
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.01)

class ADMM_network_without_sizeConstraint(ADMM_networks):

    def __init__(self, neural_network, lamda=1, sigma=0.02, kernelsize=7):
        super().__init__(neural_network, lamda=lamda, sigma=sigma, kernelsize=kernelsize, lowerbound=0, upperbound=0, )

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
        # plt.gray()
        plt.clf()
        plt.subplot(1, 1, 1)
        plt.imshow(self.uimage[0].cpu().data.numpy().squeeze(), cmap='gray')
        # plt.imshow(self.gamma[0])
        plt.contour(self.umask.squeeze().cpu().data.numpy(), level=[0], colors="yellow", alpha=0.2, linewidth=1)

        plt.contour(self.gamma[0], level=[0], colors="red", alpha=0.2, linewidth=0.001)
        plt.contour(self.heatmap2segmentation(self.uimage_output).squeeze().cpu().data.numpy(), level=[0],
                    colors="green", alpha=0.2, linewidth=0.001)
        plt.show(block=False)
        plt.pause(0.01)


class ADMM_network_without_graphcut(ADMM_networks):
    def __init__(self, neural_network, lowerbound=98, upperbound=1723):
        super().__init__(neural_network, lowerbound, upperbound, lamda=0, sigma=0, kernelsize=5)

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

