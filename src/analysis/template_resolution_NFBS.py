"""
Create the template resolution measure for the NFBS derived templates and registered brains in the affine and rigid case.
"""

import config
from tools import template_resolution

from itertools import product
import numpy as np

nfbs_path = '../data/processed/NFBS_Dataset/' # relative to src folder

for dataset, mode, norm, eff_height in product(config.nfbs_datasets,
                                                config.nfbs_modes,
                                                config.nfbs_norms,
                                                config.nfbs_eff_heights):
    print('create template resolution measure for:', dataset, mode, norm, eff_height)
            
    nfbs_reg = np.load(nfbs_path + f'{dataset}_{mode}_{norm}_registration.npy', mmap_mode='r')
    
    minimal_sig = template_resolution(nfbs_reg, eff_height=eff_height, quantile_range=config.quantile_range, sig_step=0.5, use_gpu=config.use_gpu)

    np.save(nfbs_path + f'{dataset}_{mode}_{norm}_template_resolution_eh_{eff_height}.npy', minimal_sig)

    # visualize for debugging:
    #import matplotlib.pyplot as plt
    #plt.rcParams['figure.dpi'] = 200
    #plt.imshow(minimal_sig[:,100,:])
    #plt.show()
