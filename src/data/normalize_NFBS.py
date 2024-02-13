"""
Convert NFBS brains from seperate nifti files to a single npy file
and create intensity normalized versions (by rescaling the intensities)
"""

import os
import numpy as np
from natsort import natsorted
from glob import glob
import nibabel as nib

data_path = 'data/raw/NFBS_Dataset/'  # relative to Makefile
data_path_out = 'data/processed/NFBS_Dataset/'

# create processed folder
if not os.path.exists(data_path_out):
    os.makedirs(data_path_out)


# extracted brains
NFBS_brain_nii = natsorted(glob(data_path + '*/*brain.nii.gz'))

n_nii = len(NFBS_brain_nii)
mri_shape = nib.load(NFBS_brain_nii[0]).get_fdata().shape

# convert nii.gz to numpy
# "open_memmap" can CREATE an empty npy file of a given shape and load it as memmap!
NFBS_brain = np.lib.format.open_memmap(data_path_out + 'NFBS_brain.npy',
                                       mode='w+', # w+ for creating or overwriting
                                       dtype=np.float32,
                                       shape=(n_nii, *mri_shape))
for i, nii in enumerate(NFBS_brain_nii):
    print(f'NFBS_brain.npy: converted {i + 1} of {n_nii}', end='\r')
    NFBS_brain[i, ...] = nib.load(nii).get_fdata()
print('')


# # entire heads (not used)
# NFBS_nii = natsorted(glob(data_path + '*/*w.nii.gz'))

# # convert nii.gz to numpy
# NFBS = np.lib.format.open_memmap(data_path_out + 'NFBS.npy',
#                                        mode='w+',
#                                        dtype=np.float32,
#                                        shape=(n_nii, *mri_shape))
# for i, nii in enumerate(NFBS_nii):
#     print(f'NFBS.npy: converted {i + 1} of {n_nii}', end='\r')
#     NFBS[i, ...] = nib.load(nii).get_fdata()
# print('')


# create intensity normalized images by simple rescaling:
#   [0, median, maximum] -> [0, 0.5, maximum / median / 2]
# the medians are computed *without* the background zeros!
NFBS_brain_norm = np.lib.format.open_memmap(data_path_out + 'NFBS_brain_norm.npy',
                                       mode='w+',
                                       dtype=np.float32,
                                       shape=(n_nii, *mri_shape))
for i, brain in enumerate(NFBS_brain):
    print(f'NFBS_brain_norm.npy: rescaled {i + 1} of {n_nii}', end='\r')
    median = np.median(brain[brain > 0])
    NFBS_brain_norm[i, ...] = brain / median / 2
print('')


# # normalization for entire heads
# # NFBS_norm = np.empty(shape=(n_nii, *mri_shape), dtype=np.float32)
# # for i, brain in enumerate(NFBS):
# #     print(f'NFBS_norm.npy: rescaled {i + 1} of {n_nii}', end='\r')
# #     median = np.median(brain[brain > 0])
# #     NFBS_norm[i, ...] = brain / median / 2
# # print('')
# # np.save(data_path_out + 'NFBS_norm.npy', NFBS_norm)
# # does not work well for whole brains, just rescale [0, 99%-quantile] -> [0, 1]
# print('create NFBS_norm.npy ...')
# q99 = np.quantile(NFBS, 0.99)
# np.save(data_path_out + 'NFBS_norm.npy', NFBS / q99)
