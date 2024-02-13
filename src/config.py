'''
Set global parameters:
'''

from os import environ as env


# template generation
max_iter = 200
toy_data_max_iter = 200


# template resolution measure
use_gpu = ('1' == env.get('TR_USE_GPU')) # set in Makefile
quantile_range = [0.1, 0.9]


# NFBS dataset parameters for template resultion measure and visualizations
nfbs_datasets = ['NFBS_brain'] # 'NFBS' for whole head not used
nfbs_modes = ['rigid', 'affine']
nfbs_norms = ['l2', 'l1']
nfbs_eff_heights = [0.3, 0.4, 0.5, 0.6]
nfbs_slice = 95 # index of brain slices in horizontal plane for visualization
nfbs_sig_max = 7.0 # max sigma for visualizations


# MNIST dataset parameters for template resultion measure and visualizations
mnist_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
mnist_modes = ['rigid', 'affine']
mnist_norms = ['l2', 'l1']
mnist_eff_heights = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
mnist_sig_max = 1.8 # max sigma for visualizations (only digit 3)


# 1d toy dataset parameters for template resultion measure and visualizations
toy_data_modes = ['rigid', 'affine']
toy_data_norms = ['l2', 'l1']
toy_data_eff_heights = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
