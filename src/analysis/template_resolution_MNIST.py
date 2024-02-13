"""
Create the template resolution measure for the MNIST derived templates and registered images in the affine and rigid case.
"""

import config
from tools import template_resolution

from itertools import product
import numpy as np

mnist_path = '../data/processed/MNIST/' # relative to src folder

for digit, mode, norm, eff_height in product(config.mnist_digits,
                                                config.mnist_modes,
                                                config.mnist_norms,
                                                config.mnist_eff_heights):
    print('create template resolution measure for:', digit, mode, norm, eff_height)

    mnist_reg = np.load(mnist_path + f'digit_{digit}_{mode}_{norm}_registration.npy')

    minimal_sig = template_resolution(mnist_reg, eff_height=eff_height, quantile_range=config.quantile_range, sig_step=0.01, use_gpu=config.use_gpu)

    np.save(mnist_path + f'digit_{digit}_{mode}_{norm}_template_resolution_eh_{eff_height}.npy', minimal_sig)

    # # visualize for debugging:
    # import matplotlib.pyplot as plt
    # plt.rcParams['figure.dpi'] = 200
    # plt.imshow(minimal_sig)
    # plt.show()

    # test alterantive quantile_range
    minimal_sig_2 = template_resolution(mnist_reg, eff_height=eff_height, quantile_range=[0.15, 0.85], sig_step=0.01, use_gpu=config.use_gpu)
    np.save(mnist_path + f'digit_{digit}_{mode}_{norm}_template_resolution_eh_{eff_height}_qr70.npy', minimal_sig_2)
