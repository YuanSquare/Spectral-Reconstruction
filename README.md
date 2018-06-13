# Spectral-Reconstruction
PyTorch implementation of Spectral-reconstruction from dispersive blurring Convolutional Neural Network 
## l8_rgb_blur.py  
Keras impementation. 

INPUT: A blurred image(single channel) and a clear RGB image(3 channels) ;

OUTPUT: Spectral data cube (16 channels, adjustable  in different cases)

## resnet_mse_2inflow_pooling_gap.py
PyTorch impementation.

   The model corresponds to acquisition system(including optical components, CCD,lens and so on ). There are two inflows: one get clear grayscale texture, the orther acquires is  blurred image which is caused by a prism dispersion (it is the shift of spectral data addition), a multi-scale information flow was adopted to expand the receptive field, the other is responsible to extract geometric features. 

   Globle Average Pooling is used after 16 times downsampling(to take the place of fully connected layer ,for when you uses the Fully connected layer, the size of input cannot be changed anymore.)
