# Spectral-Reconstruction
PyTorch implementation of Spectral-reconstruction from dispersive blurring Convolutional Neural Network 
## l8_rgb_blur.py  
Keras impementation. 

INPUT: A blurred image(single channel) and a clear RGB image(3 channels) ;

OUTPUT: Spectral data cube (16 channels, adjustable  in different cases)

The model is very simple which contains 8 convolutional layers.

## resnet_mse_2inflow_pooling_gap.py
PyTorch impementation.

   The model corresponds to acquisition system(including optical components, CCD,lens and so on ). There are two inflows: one get clear grayscale texture, the orther acquires is  blurred image which is caused by a prism dispersion (it is the shift of spectral data addition), a multi-scale information flow was adopted to expand the receptive field, the other is responsible to extract geometric features. 

   Globle Average Pooling is used after 16 times downsampling(to take the place of fully connected layer ,for when you uses the Fully connected layer, the size of input cannot be changed anymore.)
   
## resnext_grad_mse_naya.py
   
  Based on the inspiration of  inception structure in GoogleNet, the model unit has been further improved. The muti-RF structure contains 5 branchs which has different RF to get more information from different scale and kernal size.
  
  Another contribution is that i put forward the mean square error function of spectral latitude gradient which can change the gradient descent direction during training and further restrict the trend information of spectral data in spectral latitude
