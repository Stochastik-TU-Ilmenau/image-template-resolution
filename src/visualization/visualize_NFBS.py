"""
Create plots for the NFBS dataset.
"""

import config

from itertools import product
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

nfbs_path = '../data/processed/NFBS_Dataset/'  # relative to src folder
plot_path = '../plots/NFBS_Dataset/'

# create plots folder
import os
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

slice = config.nfbs_slice

for dataset, mode, norm in product(config.nfbs_datasets, config.nfbs_modes, config.nfbs_norms):

    try:
        template = np.load(nfbs_path + f'{dataset}_{mode}_{norm}_template.npy')
        img_reg = np.load(nfbs_path + f'{dataset}_{mode}_{norm}_registration.npy', mmap_mode='r')
    except:
        continue

    # plot template
    plt.imshow(template[:, slice, :], cmap='Greys_r', vmin=0)
    plt.colorbar()
    plt.savefig(plot_path + f'{dataset}_{mode}_{norm}_template.png', dpi=300)
    plt.show(); plt.clf()
    
    # plot standard deviation
    plt.imshow(np.std(img_reg, axis=0)[:, slice, :])
    plt.colorbar()
    plt.savefig(plot_path + f'{dataset}_{mode}_{norm}_std.png', dpi=300)
    plt.show(); plt.clf()

    for eff_height in config.nfbs_eff_heights:
        print('create plots for:', dataset, mode, norm, eff_height)

        try:
            minimal_sig = np.load(nfbs_path + f'{dataset}_{mode}_{norm}_template_resolution_eh_{eff_height}.npy')
        except:
            continue

        # plot pixelwise template resolution
        plt.imshow(minimal_sig[:, slice, :], cmap=plt.colormaps['OrRd'], vmin=0, vmax=config.nfbs_sig_max)
        plt.colorbar()
        plt.savefig(plot_path + f'{dataset}_{mode}_{norm}_template_resolution_eh_{eff_height}.png', dpi=300)
        plt.show(); plt.clf()

        ## plot variation orthogonal to levelsets / along gradients ################################

        # smoothed image gradient
        sig_grad = 2.0
        tmp_grad_0 = gaussian_filter(template, sigma=sig_grad, order=(1, 0, 0))
        tmp_grad_1 = gaussian_filter(template, sigma=sig_grad, order=(0, 1, 0))
        tmp_grad_2 = gaussian_filter(template, sigma=sig_grad, order=(0, 0, 1))
        tmp_grad = np.stack([tmp_grad_0, tmp_grad_1, tmp_grad_2])

        n, _, m = template.shape
        from matplotlib import cm, colors
        col = plt.colormaps['OrRd']

        plt.imshow(template[:, slice, :], cmap='Greys_r', vmin=0) # , interpolation='bicubic')
        # plt.colorbar()
        plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(0.0, config.nfbs_sig_max), cmap=col), ax=plt.gca())
        # plt.imshow(minimal_filter_q)
        for i in np.arange(0, n, 3):
            for j in np.arange(0, m, 3):
                # print(i, j)
                g0, g1, g2 = tmp_grad[:, i, slice, j]
                nrm = np.sqrt(g0**2 + g1**2 + g2**2)
                n0, n2 = g0 / nrm, g2 / nrm
                scale = np.sqrt(n0**2 + n2**2)
                sq = minimal_sig[i, slice,  j] * scale
                cc = sq / scale / minimal_sig[:, slice, :].max()
                plt.plot(
                    [j - n2 * sq, j + n2 * sq],
                    [i - n0 * sq, i + n0 * sq],
                    alpha=1.0, lw=0.9, c=col(cc),
                    )
        plt.savefig(plot_path + f'{dataset}_{mode}_{norm}_template_res_bars_eh_{eff_height}.png', dpi=300)
        plt.show(); plt.clf()


# check histogram of template
template = np.load(nfbs_path + f'NFBS_brain_affine_l2_template.npy')
plt.hist(template[template > 0.05].ravel(), bins=100)
plt.xlim(0, 1)
plt.savefig(plot_path + f'NFBS_brain_affine_l2_template_hist.png', dpi=300)
plt.show(); plt.clf()

# plot sample from registered brains
img = np.load(nfbs_path + f'NFBS_brain.npy', mmap_mode='r')
img_reg = np.load(nfbs_path + f'NFBS_brain_affine_l2_registration.npy', mmap_mode='r')
n_sample = 10 # arbitrary sample
plt.imshow(img_reg[n_sample, :, slice, :], cmap='Greys_r')
plt.colorbar()
plt.savefig(plot_path + f'NFBS_brain_affine_l2_sample.png', dpi=300)
plt.show(); plt.clf()

plt.imshow(img[n_sample, :, slice, :], cmap='Greys_r')
plt.colorbar()
plt.savefig(plot_path + f'NFBS_brain_sample.png', dpi=300)
plt.show(); plt.clf()
