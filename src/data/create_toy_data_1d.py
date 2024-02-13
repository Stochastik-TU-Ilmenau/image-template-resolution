"""
Create a toy data set of 1d signals for illustration purposes (not used in paper)
"""

import os
import numpy as np

data_path = 'data/raw/toy_data_1d/'  # relative to Makefile

# create processed folder
if not os.path.exists(data_path):
    os.makedirs(data_path)



## create 100 signals of two bumps with different horizontal shifts

# using date as seed
np.random.seed(20230511)

data = np.zeros((100, 1000))

def rand_shift(sig, scale=10):
    shift = int(scale * np.random.randn())
    return np.roll(sig, shift)

data[:, 200:300] = 1.0
data = np.apply_along_axis(rand_shift, axis=1, arr=data, scale=25)

data[:, 500:800] = 0.9
data = np.apply_along_axis(rand_shift, axis=1, arr=data, scale=25)

from scipy.ndimage import gaussian_filter1d

data = gaussian_filter1d(data, sigma=5, mode='constant')
data = data + 0.005 * np.random.randn(100, 1000)

import matplotlib.pyplot as plt
plt.plot(data.T)
plt.show()

np.save(data_path + 'toy_data_1d_signals.npy', data)
