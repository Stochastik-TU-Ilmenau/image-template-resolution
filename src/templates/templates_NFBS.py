"""
Create affine and rigid template/atlas for the 3D normalized NFBS dataset and save the registered brains.
"""

import config
from tools import registration

from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import torch as t
t.set_default_dtype(t.float32)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')


nfbs_path = '../data/processed/NFBS_Dataset/' # relative to src folder

for dataset in config.nfbs_datasets:

    # using normalized versions for registration!
    NFBS_x = np.load(nfbs_path + f'{dataset}_norm.npy', mmap_mode='r+')

    try:
        nfbs = t.as_tensor(NFBS_x, device=device) # cuda if enough memory
    except:
        nfbs = t.as_tensor(NFBS_x) # cpu

    def plot_template(template):
        plt.imshow(template.detach().cpu().numpy()[:, 100, :])
        plt.show()
        plt.imshow(template.detach().cpu().numpy()[110, :, :])
        plt.show()
        plt.imshow(template.detach().cpu().numpy()[:, :, 90].T)
        plt.show()

    for mode, norm in product(config.nfbs_modes, config.nfbs_norms):

        # perform registration:
        template, affine_maps, nfbs_reg, intesity_rescale = registration(nfbs, mode=mode, max_iter=config.max_iter, data_norm=norm, scale_intensities=True, plot_template=plot_template)

        # save results:
        print('save results ...')
        np.save(nfbs_path + f'{dataset}_{mode}_{norm}_registration.npy', nfbs_reg)
        np.save(nfbs_path + f'{dataset}_{mode}_{norm}_template.npy', template)
        np.save(nfbs_path + f'{dataset}_{mode}_{norm}_maps.npy', affine_maps)
        np.save(nfbs_path + f'{dataset}_{mode}_{norm}_int_rescales.npy', intesity_rescale)
