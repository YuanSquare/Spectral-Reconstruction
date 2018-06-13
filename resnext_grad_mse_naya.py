# import os
# # # #using GPU 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from logger import Logger
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

from torch.utils.data import Dataset, DataLoader

# hyper parameters
BATCH_SIZE = 32

def GRAD_MSE(y_true,y_pre):
    filters = torch.FloatTensor([-1., 1.]).resize_(1, 1, 2, 1, 1)
    # cha = y_true - y_pre
    # cha = cha.unsqueeze(1)
    # res = Variable(torch.zeros(cha.shape)).cuda()
    g_gt = torch.nn.functional.conv3d(y_true.unsqueeze(1).cuda(), Variable(filters).cuda(), stride=1, padding=0)
    g_pre = torch.nn.functional.conv3d(y_pre.unsqueeze(1).cuda(), Variable(filters).cuda(), stride=1, padding=0)
    cha = g_gt - g_pre
    # print "res:", res.size()
    res = cha**2
    grad_mse = torch.mean(res)
    # print "t:", grad_mse.size()
    return grad_mse

def psnr(y_true, y_pred):
    mse = torch.mean((y_true - y_pred)**2)
    mse = torch.mean(mse)
    return 20 * torch.log(1. / torch.sqrt(mse)) / np.log(10)#Attention,if you normalization pixels ,you use 1 not 255

def mse_zyy(y_true,y_pred):
    mse = torch.mean((y_true - y_pred)**2)
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

class residualBlock(nn.Module):
    def __init__(self):
        super(residualBlock, self).__init__()
        self.resnext_bottle = nn.Sequential(
            nn.Conv2d(16, 4, (5, 3), 1, (2, 1)),
            nn.ReLU(),
            nn.Conv2d(4, 4, (5, 3), 1, (2, 1)),
            nn.ReLU(),
            nn.Conv2d(4, 16, (5, 3), 1, (2, 1)),
            nn.ReLU(),
        )
    def forward(self, x):
        y = self.resnext_bottle(x)
        return y + x

class residualBlock64(nn.Module):
    def __init__(self):
        super(residualBlock64, self).__init__()
        self.Bottleneck64 = nn.Sequential(
            nn.Conv2d(64, 64, (5, 3), 1, (2, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (5, 3), 1, (2, 1)),
            nn.ReLU(),
        )
    def forward(self, x):
        y = self.Bottleneck64(x)
        return y + x
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1
        self.conv_input_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
        )
        # 2
        self.conv_input_2 = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 3), 1, (2, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(128, 256, (5, 3), 1, (2, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        for j in range(10):
            for i in range(16):
                self.add_module('residual_block' + str(j + 1) + str(i + 1), residualBlock())

        self.conv_up1 = nn.Sequential(
        nn.Conv2d(256, 1024, (5, 3), 1, (2, 1)),
        nn.ReLU(),
        nn.PixelShuffle(2),
        )

        self.conv_up2 = nn.Sequential(
            nn.Conv2d(256, 256, (5, 3), 1, (2, 1)),
            nn.ReLU(),
            nn.PixelShuffle(2),
        )   # out feature: 64


        for i in range(8):
            self.add_module('residual_Block' + str(i + 1), residualBlock64())

        self.Bottleneck64 = nn.Sequential(
            nn.Conv2d(64, 64, (5, 3), 1, (2, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (5, 3), 1, (2, 1)),
            nn.ReLU(),
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x):

        x_1 = x[:, 0, :x.size(2)-16, :].unsqueeze(0).permute(1, 0, 2, 3)  # gray

        # print "x1", x_1.size()
        x_1 = self.conv_input_1(x_1)

        x_1 = self.conv1_1(x_1)

        x_2 = x

        x_2 = self.conv_input_2(x_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.conv_p1(x_2)
        x_2 = self.conv_p2(x_2)

        # print x_2.size()
        x_2 = f.adaptive_max_pool2d(x_2, x_1.size(3)/4)  # 64 64

        for j in range(10):  # depth
            for i in range(16):  # group
                x_3 = x_2[:, i*16:(i+1)*16, :, :]   # width
                # print "******", x_3.size()
                x_4 = self.__getattr__('residual_block' + str(j + 1) + str(i + 1),)(x_3)
                if i == 0:
                    x_5 = x_4
                else:
                    x_5 = torch.cat((x_5, x_4), 1)
            x_2 = x_5 + x_2

        x_2 = self.conv_up1(x_2)
        x_2 = self.conv_up2(x_2)

        x = x_2 + x_1

        for i in range(8):
            # x1 = self.Bottleneck64(x)
            # x = x1 + x
            x = self.__getattr__('residual_Block' + str(i + 1), )(x)

        x_out = self.conv_out(x)
        return x_out

LR = 1e-4
Resnext_kernal = CNN().cuda()
optimizer = torch.optim.Adam(Resnext_kernal.parameters())   # optimize all cnn parameters
loss_func = nn.MSELoss()

## dataset
class MyDataset1(Dataset):
    def __init__(self, data_file, lable_file):
        self.file_data = h5py.File(str(data_file), 'r')
        self.lable_file = h5py.File(str(lable_file), 'r')

    def __len__(self):
        return 60264

    def __getitem__(self, idx):
        img = self.file_data['test'][idx, :, :, :]
        img = torch.FloatTensor(img)

        img_label = self.lable_file['test'][idx, :, :64, :]
        img_label = torch.FloatTensor(img_label)
        sample = {'img_dis': img, 'img_label': img_label}
        return sample

class MyDataset2(Dataset):
    def __init__(self, data_file, lable_file):
        self.file_data = h5py.File(str(data_file), 'r')
        self.lable_file = h5py.File(str(lable_file), 'r')

    def __len__(self):
        return 87534

    def __getitem__(self, idx):
        img = self.file_data['test'][idx, :, :, :]
        img = torch.FloatTensor(img)

        img_label = self.lable_file['test'][idx, :, :64, :]
        img_label = torch.FloatTensor(img_label)
        sample = {'img_dis': img, 'img_label': img_label}
        return sample

class MyDataset3(Dataset):
    def __init__(self, data_file, lable_file):
        self.file_data = h5py.File(str(data_file), 'r')
        self.lable_file = h5py.File(str(lable_file), 'r')

    def __len__(self):
        return 13985

    def __getitem__(self, idx):
        img = self.file_data['test'][idx, :, :, :]
        img = torch.FloatTensor(img)

        img_label = self.lable_file['test'][idx, :, :64, :]
        img_label = torch.FloatTensor(img_label)
        sample = {'img_dis': img, 'img_label': img_label}
        return sample


dataset1 = MyDataset1(data_file='./h5_train/spec_train_in_64_80_1.h5',
                    lable_file='./h5_train/spec_train_out_64_80_1.h5',)
dataset2 = MyDataset2(data_file='./h5_train/spec_train_in_64_80_2.h5',
                    lable_file='./h5_train/spec_train_out_64_80_2.h5',)
dataset3 = MyDataset3(data_file='./h5_train/spec_test_in_64_80_2.h5',
                    lable_file='./h5_train/spec_test_out_64_80_2.h5',)
# Data.ConcatDataset([dataset1, dataset2])
loader = Data.DataLoader(dataset=Data.ConcatDataset([dataset1, dataset2]),
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         # num_workers=2,
                         )
loader1 = Data.DataLoader(dataset=dataset3,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         # num_workers=2,
                         )
# file_train = h5py.File('./h5_train/spec_train_in_64_80_1.h5', 'r')
# train_data = torch.FloatTensor(file_train['test'][:])
# # print train_data.shape
# file_train = h5py.File('./h5_train/spec_train_in_64_80_1.h5', 'r')
# train_label = torch.FloatTensor(file_train['test'][:, :, :64, :])
# train_dataset1 = Data.TensorDataset(data_tensor=train_data, target_tensor=train_label)
# #
file_train = h5py.File('./train6/naya_train_in_60_76.h5', 'r')
train_data = torch.FloatTensor(file_train['test'][:])
file_train = h5py.File('./train6/naya_train_in_60_76.h5', 'r')
train_label = torch.FloatTensor(file_train['test'][:])

train_dataset = Data.TensorDataset(data_tensor=train_data, target_tensor=train_label)

loader = Data.DataLoader(dataset=train_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         # num_workers=4,
                         )
file_test = h5py.File('./h5_train/naya_test_in_60_76.h5', 'r')
test_data = torch.FloatTensor(file_test['test'][:])
# print train_data.shape
file_test = h5py.File('./h5_train/naya_test_out_60_76.h5', 'r')
test_label = torch.FloatTensor(file_test['test'][:])
test_dataset = Data.TensorDataset(data_tensor=test_data, target_tensor=test_label)
loader1 = Data.DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True,
                         # num_workers=4,
                         )




# Resnext_kernal = torch.load('./result/gray_blur/resnext_mse/model_grad/resnext_mse_naya_49_0.00055485_0.00111911.pkl')
loss = []
loss_test = []

logger = Logger('./result/gray_blur/resnext_mse/model_all_grad/logs')
lamda = 0.0
for epoch in range(800):
    # train
    Resnext_kernal.train()
    psnr_train = 0
    ssim_train = 0
    loss_all_batch_train = 0
    loss_grad_batch_train = 0
    for step, sample in enumerate(loader):
        batch_x, batch_y = sample['img_dis'], sample['img_label']
        # batch_x1 = batch_x.numpy()
        # lable = batch_y.numpy()
        # print "out:", batch_x1.shape, "lable:", lable.shape
        #
        # img = batch_x1[:, 0, :64, :].squeeze(0)
        # print img.shape
        # #
        # plt.imshow(img, cmap=plt.get_cmap("gray"))
        # plt.imshow(np.array(img))
        # plt.show()
        #
        # img = lable[:, 10, :, :].squeeze(0)
        # plt.imshow(img, cmap=plt.get_cmap("gray"))
        # plt.imshow(np.array(img))
        # plt.show()

        batch_x, batch_y = Variable(batch_x).cuda(), Variable(batch_y).cuda()

        pre = Resnext_kernal(batch_x)
        loss_batch1 = loss_func(pre, batch_y)
        loss_batch2 = GRAD_MSE(pre, batch_y)

        loss_batch = loss_batch1 + lamda * loss_batch2
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        psnr_ = psnr(pre, batch_y)
        psnr_train += psnr_.data[0]
        ssim_ = ssim(pre, batch_y)
        ssim_train += ssim_.data[0]
        loss_all_batch_train = loss_all_batch_train + loss_batch1.data[0]
        loss_grad_batch_train = loss_grad_batch_train + loss_batch2.data[0]

        print 'step:', step, 'Epoch:', epoch, '|train_mse_loss:%.7f ' % loss_batch1.data[0], '|train_grad_mse_loss:%.7f ' % loss_batch2.data[0], 'PSNR:', psnr_.data[0], 'SSIM:', ssim_.data[0], 'loss_all:', loss_batch.data[0]
        # print 'step:', step, 'Epoch:', epoch, '|train_mse_loss:%.7f ' % loss_batch1.data[0], 'PSNR:', psnr_.data[0], 'SSIM:', ssim_.data[0], 'loss_all:', loss_batch.data[0]

    psnr_train = psnr_train / step
    ssim_train = ssim_train / step
    loss_all_batch_train = loss_all_batch_train / step
    loss_grad_batch_train = loss_grad_batch_train / step

    loss.append(loss_all_batch_train)
    # test
    Resnext_kernal.eval()
    loss_all_batch_test = 0
    psnr_val = 0
    ssim_val = 0
    for step1, sample in enumerate(loader1):
        batch_x_val, batch_y_val = sample['img_dis'], sample['img_label']
        batch_x_val, batch_y_val = Variable(batch_x_val, volatile=True).cuda(), Variable(batch_y_val, volatile=True).cuda()
        pre_val = Resnext_kernal(batch_x_val)
        loss_batch_val = loss_func(pre_val, batch_y_val)
        loss_all_batch_test = loss_all_batch_test + loss_batch_val.data[0]
        psnr__ = psnr(pre_val, batch_y_val)
        psnr_val = psnr_val+psnr__.data[0]
        ssim__ = ssim(pre_val, batch_y_val)
        ssim_val = ssim_val + ssim__.data[0]
        print 'step:', step1, 'Epoch:', epoch, '|test_loss:%.7f ' % loss_batch_val.data[0], 'PSNR:', psnr__.data[0], 'SSIM:', ssim__.data[0]
    loss_all_batch_test = loss_all_batch_test/step1
    psnr_val = psnr_val / step1
    ssim_val = ssim_val / step1
    loss_test.append(loss_all_batch_test)
    np.save("./result/gray_blur/resnext_mse/model_all_grad/resnextloss.npy", loss)
    np.save("./result/gray_blur/resnext_mse/model_all_grad/resnextloss.npy", loss_test)
    print 'Epoch:', epoch, '|train_loss :%.7f ' % loss_all_batch_train, '|test_loss :%.7f ' % loss_all_batch_test, '|psnr_train :%.7f ' % psnr_train, '|psnr_val :%.7f ' % psnr_val
    torch.save(Resnext_kernal, './result/gray_blur/resnext_mse/model_all_grad/resnext_mse_naya_%d_%.8f_%.8f.pkl' % (epoch, loss_all_batch_train, loss_all_batch_test))

    info = {
        'train_loss': loss_all_batch_train,
        'loss_grad_train': loss_grad_batch_train,
        'test_loss': loss_all_batch_test,
        'psnr_train': psnr_train,
        'psnr_val': psnr_val,
        'ssim_train': ssim_train,
        'ssim_val': ssim_val,

    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)
