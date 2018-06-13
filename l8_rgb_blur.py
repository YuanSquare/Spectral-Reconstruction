import os
#using GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
config = tf.ConfigProto()
#use 80% of the GPU memory
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)

from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras import optimizers
import numpy
import math
import h5py


def psnr(y_true, y_pred):
    mse = K.mean(K.square(y_true[:,:,0] - y_pred[:,:,0]), axis=(-3, -2))
    mse = K.mean(mse)
    return 20 * K.log(1. / K.sqrt(mse)) / numpy.log(10)#Attention,if you normalization pixels ,you use 1 not 255

def mse_zyy(y_true,y_pred):
    mse = K.mean(K.square(y_true[:,:,0] - y_pred[:,:,0]), axis=(-3, -2))
    mse= K.mean(mse)
    return mse

def ssim(y_true, y_pred):
    K1 = 0.01
    K2 = 0.03
    mu_x = K.mean(y_pred)
    mu_y = K.mean(y_true)
    sig_x = K.std(y_pred)
    sig_y = K.std(y_true)
    sig_xy = K.mean(y_pred * y_true) - mu_x * mu_y
    #sig_xy = (sig_x * sig_y) ** 0.5
    L =  1
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    ssim = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2) * 1.0 / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    return ssim

def create_model(img_height,img_width,img_channel):
    ip = Input(shape=(img_height, img_width,img_channel))
    L1 = Conv2D(32, (11, 11), padding='same', activation='relu', kernel_initializer='glorot_uniform')(ip)
    L2 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L1)
    L3 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L2)
    L4 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L3)
    L4=concatenate([L4,L1],axis=-1)
    L5 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L4)
    L6 = Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L5)
    L6=concatenate([L6,L1],axis=-1)
    L7 = Conv2D(128, (1, 1), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L6)
    L8 = Conv2D(16, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform')(L7)
    deblocking =Model(inputs=ip,outputs= L8)
    optimizer = optimizers.Adam(lr=1e-4)
    deblocking.compile(optimizer=optimizer,loss='mean_squared_error', metrics=[psnr,ssim,mse_zyy])
    return deblocking

'''
from keras.utils import plot_model
plot_model(deblocking, to_file='model1.png', show_shapes=True, show_layer_names=True)
'''
def main():
    file_train = h5py.File('./train_rgb_blur/naya_train_in_60_76.h5','r')
    train_data = file_train['test'][:].transpose(0,3,2,1)
    file_train1 = h5py.File('./train_rgb_blur/naya_train_out_60_76.h5','r')
    train_label = file_train1['test'][:].transpose(0,3,2,1)

    file_test = h5py.File('./train_rgb_blur/naya_test_in_60_76.h5','r')
    test_data = file_test['test'][:].transpose(0,3,2,1)
    file_test1 = h5py.File('./train_rgb_blur/naya_test_out_60_76.h5','r')
    test_label = file_test1['test'][:].transpose(0,3,2,1)

    fvc_model = create_model(60,76,4)
    checkpoint=ModelCheckpoint(filepath='./result/rgb_blur/models/{epoch:02d}-{val_loss:.7f}.hdf5',period=1)
    fvc_model.fit(train_data, train_label,
                    epochs=200,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(test_data, test_label),callbacks=[checkpoint])

if __name__ == "__main__":
    main()
