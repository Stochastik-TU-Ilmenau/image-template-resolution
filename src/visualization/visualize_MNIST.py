"""
Create plots for the MNIST dataset.
"""

import config

from itertools import product
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

mnist_path = '../data/processed/MNIST/' # relative to src folder
plot_path = '../plots/MNIST/'

# create plots folder
import os
if not os.path.exists(plot_path):
    os.makedirs(plot_path)


for digit, mode, norm in product(config.mnist_digits, config.mnist_modes, config.mnist_norms):

    try:
        template = np.load(mnist_path + f'digit_{digit}_{mode}_{norm}_template.npy')
        mnist_reg = np.load(mnist_path + f'digit_{digit}_{mode}_{norm}_registration.npy')
    except:
        continue

    # plot template
    plt.imshow(template, cmap='Greys', vmin=0)
    plt.colorbar()
    plt.savefig(plot_path + f'digit_{digit}_{mode}_{norm}_template.png', dpi=300)
    plt.show(); plt.clf()

    for eff_height in config.mnist_eff_heights:
        print('create plots for:', digit, mode, norm, eff_height)
    
        try:
            minimal_sig = np.load(mnist_path + f'digit_{digit}_{mode}_{norm}_template_resolution_eh_{eff_height}.npy')
            minimal_sig_2 = np.load(mnist_path + f'digit_{digit}_{mode}_{norm}_template_resolution_eh_{eff_height}_qr70.npy')
        except:
            continue

        vmax = config.mnist_sig_max if digit == 3 else None
        
        # plot pixelwise template resolution
        plt.imshow(minimal_sig, cmap=plt.colormaps['OrRd'], vmin=0, vmax=vmax)
        if digit == 3 and norm == 'l1':
            plt.axhline(17, ls='--', c='k', lw=1)
        plt.colorbar()
        plt.savefig(plot_path + f'digit_{digit}_{mode}_{norm}_template_resolution_eh_{eff_height}.png', dpi=300)
        plt.show(); plt.clf()

        # plot pixelwise template resolution for alternative quantile_range
        plt.imshow(minimal_sig_2, cmap=plt.colormaps['OrRd'], vmin=0, vmax=vmax)
        plt.colorbar()
        plt.savefig(plot_path + f'digit_{digit}_{mode}_{norm}_template_resolution_eh_{eff_height}_qr70.png', dpi=300)
        plt.show(); plt.clf()

        ## plot variation orthogonal to levelsets / along gradients ################################

        sig_grad = 1.0
        tmp_grad_0 = gaussian_filter(template, sigma=sig_grad, order=(1, 0))
        tmp_grad_1 = gaussian_filter(template, sigma=sig_grad, order=(0, 1))
        tmp_grad = np.stack([tmp_grad_0, tmp_grad_1])

        # gradient based on registered images
        # sig_grad = 1.0
        # tmp_grad_0 = gaussian_filter(mnist_reg, sigma=(0, sig_grad, sig_grad), order=(0, 1, 0))
        # tmp_grad_1 = gaussian_filter(mnist_reg, sigma=(0, sig_grad, sig_grad), order=(0, 0, 1))
        # tmp_grad = np.stack([tmp_grad_0, tmp_grad_1])
        # tmp_grad = tmp_grad.mean(axis=1)

        n, m = template.shape
        from matplotlib import cm, colors
        col = plt.colormaps['OrRd'] # 'Reds', 'OrRd', 'RdPu'

        plt.imshow(template, cmap='Greys', vmin=0)#, interpolation='bicubic')
        vmax = config.mnist_sig_max if digit == 3 else minimal_sig.max()
        plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(0.0, vmax), cmap=col), ax=plt.gca())
        #plt.imshow(minimal_filter_q)

        # sort pixel coordinates by the magnitude of minimal_sig:
        ij_sorted = np.unravel_index(np.argsort(minimal_sig, axis=None), (n, m))

        #for i in np.arange(0, n, 1):
        #    for j in np.arange(0, m, 1):
        for i, j in zip(*ij_sorted):
            #print(i, j)
            gi, gj = tmp_grad[:, i, j]
            nrm = np.sqrt(gi**2 + gj**2)
            ni, nj = gi / nrm, gj / nrm
            scale = 1.0 # 2 * minimal_sig.max()
            sq = minimal_sig[i, j] / scale # TODO: how to scale length?
            cc = sq * scale / minimal_sig.max() # color(length)
            plt.plot(
                [j - nj * sq, j + nj * sq],
                [i - ni * sq, i + ni * sq],
                alpha=1.0, lw=1.5, c=col(cc),
                solid_capstyle='round',
                )
            # additional thin black line for visual support?
            #plt.plot(
            #    [j - nj * sq, j + nj * sq],
            #    [i - ni * sq, i + ni * sq],
            #    alpha=1.0, lw=0.1, c='k',
            #    )
        plt.savefig(plot_path + f'digit_{digit}_{mode}_{norm}_template_res_bars_eh_{eff_height}.png', dpi=300)
        plt.show(); plt.clf()



# create 1d slices for digit 3
plt.style.use('seaborn-v0_8-paper') # default, ggplot, seaborn-v0_8-paper
plt.rcParams.update({
    "text.usetex": True,
    #'text.latex.preamble': r"\usepackage{amsmath}",
    "font.family": "Helvetica",
    "font.size": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

def slice_plot(template, mnist_reg, minimal_sig, slice, file_name):
    plt.plot(mnist_reg[:, slice, :].T, c='darkslategrey', alpha=0.08)
    plt.plot([], c='darkslategrey', alpha=0.4, label=r'$\mathrm{samples}$')
    plt.plot(template[slice, :], c='darkslategrey', label=r'$\mathrm{template}$')
    plt.plot([], c='orangered', label='$\sigma^*$')
    plt.ylim(-0.05, 1.3)
    plt.legend()
    ax2 = plt.gca().twinx()
    ax2.plot(minimal_sig[slice, :], c='orangered')
    ax2.tick_params(axis='y', labelcolor='orangered')
    ax2.set_ylim(-0.05, 1.3)
    plt.savefig(plot_path + file_name, dpi=300, bbox_inches='tight')
    plt.show(); plt.clf()

# rigid
template = np.load(mnist_path + f'digit_3_rigid_l1_template.npy')
mnist_reg = np.load(mnist_path + f'digit_3_rigid_l1_registration.npy')
minimal_sig = np.load(mnist_path + f'digit_3_rigid_l1_template_resolution_eh_0.6.npy')

#slice_plot(template, mnist_reg, minimal_sig, slice=7, file_name='digit_3_slice_rigid_1.pdf')
slice_plot(template, mnist_reg, minimal_sig, slice=17, file_name='digit_3_slice_rigid_2.pdf')


# affine
template = np.load(mnist_path + f'digit_3_affine_l1_template.npy')
mnist_reg = np.load(mnist_path + f'digit_3_affine_l1_registration.npy')
minimal_sig = np.load(mnist_path + f'digit_3_affine_l1_template_resolution_eh_0.6.npy')

#slice_plot(template, mnist_reg, minimal_sig, slice=7, file_name='digit_3_slice_affine_1.pdf')
slice_plot(template, mnist_reg, minimal_sig, slice=17, file_name='digit_3_slice_affine_2.pdf')
