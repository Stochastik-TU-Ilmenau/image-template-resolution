"""
Create the template resolution measure for the 1D toy dataset derived templates and registered images in the affine and rigid case.
"""

import config
from tools import template_resolution

from itertools import product
import numpy as np

data_path = '../data/processed/toy_data_1d/' # relative to src folder


for mode, norm, eff_height in product(config.toy_data_modes, config.toy_data_norms, config.toy_data_eff_heights):
    print('create template resolution measure for:', mode, norm, eff_height)
        
    data_reg = np.load(data_path + f'toy_data_1d_{mode}_{norm}_registration.npy')

    minimal_sig = template_resolution(data_reg, eff_height=eff_height, quantile_range=config.quantile_range, sig_step=0.1, use_gpu=config.use_gpu)

    np.save(data_path + f'toy_data_1d_{mode}_{norm}_template_resolution_eh_{eff_height}.npy', minimal_sig)

    # # visualize for debugging:
    # import matplotlib.pyplot as plt
    # plt.rcParams['figure.dpi'] = 200
    # plt.plot(minimal_sig)
    # plt.show()
