"""
Create affine and rigid template for the 2D MNIST dataset and save the registered images.
"""

import config
from tools import registration

from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import torch as t
t.set_default_dtype(t.float32)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

from torchvision import datasets, transforms
mnist_path = '../data/raw/' # relative to src folder
mnist_path_proc = '../data/processed/MNIST/'

# create processed folder
import os
if not os.path.exists(mnist_path_proc):
    os.makedirs(mnist_path_proc)

mnist = datasets.MNIST(mnist_path, train=True, download=False, transform=transforms.ToTensor())

labels = mnist.targets
data = mnist.data
n_samples = 100

for digit in config.mnist_digits:

    MNIST_digit = data[labels == digit][:n_samples]
    MNIST_digit = MNIST_digit / 255 # normalize intensity to range [0, 1]
    print('choosen MNIST digit:', digit, '  number of samples:', n_samples)

    mnist = t.as_tensor(MNIST_digit, device=device)


    def plot_template(template):
        plt.imshow(template.detach().cpu().numpy())
        plt.show()

    for mode, norm in product(config.mnist_modes, config.mnist_norms):

        # perform registration:
        template, affine_maps, mnist_reg, _ = registration(mnist, mode=mode, max_iter=config.max_iter, scale_intensities=False, data_norm=norm, plot_template=plot_template)

        # save results:
        print('save results ...')
        np.save(mnist_path_proc + f'digit_{digit}_{mode}_{norm}_registration.npy', mnist_reg)
        np.save(mnist_path_proc + f'digit_{digit}_{mode}_{norm}_template.npy', template)
        np.save(mnist_path_proc + f'digit_{digit}_{mode}_{norm}_maps.npy', affine_maps)
