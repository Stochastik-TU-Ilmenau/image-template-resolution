"""
Create affine template for the 1D toy dataset and save the registered signals.
"""

import config
from tools import registration

from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import torch as t
t.set_default_dtype(t.float32)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')


data_path = '../data/raw/toy_data_1d/' # relative to src folder
data_path_proc = '../data/processed/toy_data_1d/'

# create processed folder
import os
if not os.path.exists(data_path_proc):
    os.makedirs(data_path_proc)


data_1d = np.load(data_path + 'toy_data_1d_signals.npy')

data = t.as_tensor(data_1d, device=device).float()
data = data[..., None] # add dummy dimension for registration (needs 2d data)

def plot_template(template):
    plt.plot(template.detach().cpu().numpy())
    plt.show()

for mode, norm in product(config.toy_data_modes, config.toy_data_norms):

    # perform registration:
    template, affine_maps, data_reg, _ = registration(data, mode=mode, max_iter=config.toy_data_max_iter, scale_intensities=False, data_norm=norm, plot_template=plot_template, one_dim=True)

    # remove dummy dimension
    data_reg = data_reg[..., 0]
    template = template[..., 0]

    # save results:
    print('save results ...')
    np.save(data_path_proc + f'toy_data_1d_{mode}_{norm}_registration.npy', data_reg)
    np.save(data_path_proc + f'toy_data_1d_{mode}_{norm}_template.npy', template)
    np.save(data_path_proc + f'toy_data_1d_{mode}_{norm}_maps.npy', affine_maps)

    plt.plot(data_reg.T, alpha=0.1, c='k')
    plt.plot(template, c='r')
    plt.show()
