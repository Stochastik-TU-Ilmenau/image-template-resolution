"""
Create plots for the 1D toy dataset.
"""

import config
from itertools import product
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

data_path = '../data/processed/toy_data_1d/' # relative to src folder
plot_path = '../plots/toy_data_1d/'

# create plots folder
import os
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

data_raw = np.load('../data/raw/toy_data_1d/toy_data_1d_signals.npy')
# plot raw data
n = len(data_raw.T)
L = n #10.0
X = np.linspace(0.0, L, n)
plt.plot(X, data_raw.T, c='k', alpha=0.1)
plt.savefig(plot_path + f'toy_data_1d_data_raw.png', dpi=300)
plt.show(); plt.clf()

for mode, norm in product(config.toy_data_modes, config.toy_data_norms):

    try:
        data_reg = np.load(data_path + f'toy_data_1d_{mode}_{norm}_registration.npy')
        template = np.load(data_path + f'toy_data_1d_{mode}_{norm}_template.npy')
    except:
        continue
    
    # plot template with registered data
    plt.plot(X, data_reg.T, c='k', alpha=0.03)
    plt.plot(X, template, c='C3')
    plt.savefig(plot_path + f'toy_data_1d_{mode}_{norm}_template.png', dpi=300)
    plt.show(); plt.clf()

    for eff_height in config.toy_data_eff_heights:
        print('create plots for:', mode, norm, eff_height)
    
        try:
            minimal_sig = np.load(data_path + f'toy_data_1d_{mode}_{norm}_template_resolution_eh_{eff_height}.npy')
        except:
            continue

        # plot pixelwise template resolution
        plt.plot(X, data_reg.T, c='k', alpha=0.03)
        ax2 = plt.gca().twinx()
        ax2.plot(X, minimal_sig / n * L, c='C1')
        ax2.tick_params(axis='y', labelcolor='C1')
        plt.savefig(plot_path + f'toy_data_1d_{mode}_{norm}_template_resolution_eh_{eff_height}.png', dpi=300)
        plt.show(); plt.clf()
