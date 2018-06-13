# import os
# # #using GPU 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import numpy as np
import math
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import torch.utils.data as Data
import random
import torch.utils.data.sampler as Sampler
import torchvision
import matplotlib.pyplot as plt
# from torchsample.modules import ModuleTrainer
# hyper parameters
EPOCH = 300               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
LR = 1e-4             # learning rate  0.001




def psnr(y_true, y_pred):
    mse = torch.mean(torch.square(y_true[:, :, 0] - y_pred[:, :, 0]), axis=(-3, -2))
    mse = torch.mean(mse)
    return 20 * torch.log(1. / torch.sqrt(mse)) / torch.log(10)#Attention,if you normalization pixels ,you use 1 not 255

def mse_zyy(y_true,y_pred):
    mse = torch.mean(torch.square(y_true[:, :, 0] - y_pred[:, :, 0]), axis=(-3, -2))
    mse= torch.mean(mse)
    return mse

def ssim(y_true, y_pred):
    K1 = 0.01
    K2 = 0.03
    mu_x = torch.mean(y_pred)
    mu_y = torch.mean(y_true)
    sig_x = torch.std(y_pred)
    sig_y = torch.std(y_true)
    sig_xy = torch.mean(y_pred * y_true) - mu_x * mu_y
    L = 1
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2) * 1.0 / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return ssim


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1
        self.conv_input_1 = nn.Sequential(
            nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(),
        )
        self.conv1_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv1_2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
        )
        # 2
        self.conv_input_2 = nn.Sequential(
            nn.Conv2d(1, 64, (7, 7), (1, 1), (3, 3)),
            nn.ReLU(),
        )
        self.conv2_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2_2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv2_3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2_4 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(6144, 6144)  #
        self.conv2r_4 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv2r_3 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv2r_2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv2r_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.Bottleneck64_01 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.Bottleneck64_02 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.Bottleneck64_11 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.Bottleneck64_12 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.Bottleneck64_21 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.Bottleneck64_22= nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
        )

        self.conv21 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128,  3, 1, 1),
            nn.ReLU(),
        )
        self.conv22 = nn.Sequential(
            nn.Conv2d(128, 128,  3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.Bottleneck128_01 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128,  3, 1, 1),
        )
        self.Bottleneck128_02 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
        )
        self.Bottleneck128_11 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
        )
        self.Bottleneck128_12 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
        )
        self.Bottleneck128_21 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
        )
        self.Bottleneck128_22 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
        )
        self.conv_out = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 16,  3, 1, 1),
        )
        self.BN = nn.BatchNorm1d(512)
        self.GAP = nn.AdaptiveAvgPool2d(2)


    def forward(self, x):
        x_1 = x[:, 0, :, :].unsqueeze(0).permute(1, 0, 2, 3)  # gray
        # print "x1", x_1.size()
        x_1 = self.conv_input_1(x_1)
        x_11 = self.conv1_1(x_1)
        x_12 = self.conv1_2(x_11)

        x_2 = x[:, 1, :, :].unsqueeze(0).permute(1, 0, 2, 3)  # blur
        x_2 = self.conv_input_2(x_2)
        x_21 = self.conv2_1(x_2)
        # print "x_21:", x_21.shape
        x_22 = self.conv2_2(x_21)
        # print "x_22:", x_22.shape
        x_23 = self.conv2_3(x_22)
        # print "x_23:", x_23.shape
        x_24 = self.conv2_4(x_23)
        print "x_24:", x_24.shape
        x_gap = self.GAP(x_24)
        print "x_gap:", x_gap.shape
        # x_2flat = x_24.view(x_24.size(0), -1)
        # print "x_2flat:", x_2flat.shape
        # x_2fc = self.fc1(x_2flat)
        x_2fc = f.relu(x_gap)
        x_2fc = self.BN(x_2fc)
        # print "x_2fc:", x_2fc.shape

        # x_2rfc = x_2fc.view(x_24.size(), -1)
        x_r24 = f.upsample(x_2fc, [x_23.size(2), x_23.size(3)], mode='bilinear')
        x_r24 = self.conv2r_4(x_r24)
        # print "x_r24:", x_r24.shape
        x_r23 = f.upsample(x_r24, [x_22.size(2), x_22.size(3)], mode='bilinear')
        x_r23 = self.conv2r_3(x_r23)
        # print "x_r23:", x_r23.shape
        x_r22 = f.upsample(x_r23, [x_21.size(2), x_21.size(3)], mode='bilinear')
        x_r22 = self.conv2r_2(x_r22)
        # print "x_r22:", x_r22.shape
        x_r21 = f.upsample(x_r22, [x_2.size(2), x_2.size(3)], mode='bilinear')
        x_r21 = self.conv2r_1(x_r21)
        # print "x_r21:", x_r21.shape

        x = x_12 + x_r21
        x1 = f.relu(x)

        x2 = self.Bottleneck64_01(x1)
        x2 = self.Bottleneck64_02(x2)
        x1 = x1 + x2
        x1 = f.relu(x1)

        x2 = self.Bottleneck64_11(x1)
        x2 = self.Bottleneck64_12(x2)
        x1 = x1 + x2
        x1 = f.relu(x1)

        x2 = self.Bottleneck64_21(x1)
        x2 = self.Bottleneck64_22(x2)
        x1 = x1 + x2
        x1 = f.relu(x1)

        x3 = self.conv21(x1)   # 64 to 128
        x3 = self.conv22(x3)

        x4 = self.Bottleneck128_01(x3)
        x4 = self.Bottleneck128_02(x4)
        x3 = x3 + x4
        x3 = f.relu(x3)

        x4 = self.Bottleneck128_11(x3)
        x4 = self.Bottleneck128_12(x4)
        x3 = x3 + x4
        x3 = f.relu(x3)

        x4 = self.Bottleneck128_21(x3)
        x4 = self.Bottleneck128_22(x4)
        x3 = x3 + x4
        x3 = f.relu(x3)

        x_out = self.conv_out(x3)
        return x_out

Resnet_kernal = CNN().cuda()
print Resnet_kernal
optimizer = torch.optim.Adam(Resnet_kernal.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()


file_train = h5py.File('./train6/naya_train_in_60_76.h5', 'r')
file_label = h5py.File('./train6/naya_train_out_60_76.h5', 'r')

Resnet_kernal = torch.load('./result/gray_blur/resnet_mse/model/res_mse_2inflow_gap76 0.00047778.pkl')

for epoch in range(800):

    BATCH_SIZE = 32
    LENS = 60000
    LEN_list = range(LENS)
    np.random.shuffle(LEN_list)

    Resnet_kernal.train()
    loss_all_batch_train = 0
    loss_all_batch_val = 0
    shuffled_index = LEN_list
    for step in range(LENS/BATCH_SIZE-1):
        batch_x = torch.FloatTensor(torch.zeros((BATCH_SIZE, 2, 76, 60)))
        batch_y = torch.FloatTensor(torch.zeros((BATCH_SIZE, 16, 76, 60)))
        for temp in range(BATCH_SIZE):
            batch_x[temp, :] = torch.FloatTensor(file_train['test'][shuffled_index[step * BATCH_SIZE+temp], :])
            batch_y[temp, :] = torch.FloatTensor(file_label['test'][shuffled_index[step * BATCH_SIZE+temp], :])
        # print "batch_x",  batch_x.shape

        # img = batch_x[10, 1, :, :]
        # plt.imshow(img, cmap=plt.get_cmap("gray"))
        # plt.imshow(np.array(img))
        # plt.show()

        batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()
        pre = Resnet_kernal(batch_x)
        loss_batch = loss_func(pre, batch_y)
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        loss_all_batch_train = loss_all_batch_train + loss_batch.data[0]
        print 'step:', step, 'Epoch:', epoch, '|train_loss:%.7f ' % loss_batch.data[0]
    loss_all_batch_train = loss_all_batch_train / step

    print 'Epoch:', epoch, '|train_loss :%.7f ' % loss_all_batch_train
    torch.save(Resnet_kernal, './result/gray_blur/resnet_mse/model/res_mse_2inflow_gap%d %.8f.pkl' % (epoch, loss_all_batch_train))
