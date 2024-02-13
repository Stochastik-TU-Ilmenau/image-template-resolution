"""
Function for computing the (pixelwise) template resolution measure for a given template based on the corresponding registered images.
"""

import numpy as np
import torch as t
from scipy.ndimage import gaussian_filter as gaussian_filter_cpu


def template_resolution(data_reg, eff_height=0.5, quantile_range=[0.1, 0.9], sig_step=0.5, use_gpu=False):
    """Computes a smoothing sigma at every point of the template which reduces the quantile range (quantile_range) of the registered image data (dara_reg) by a given amount (eff_height * difference of quantile_range).

    Args:
        data_reg (array_like): array of nd images after registration with the template where the first axis runs over the individual images.
        eff_height (float): height of an typical (sharp) edge in the images, should be around half the image intensity range (i.e. around 0.5 for images normalized to [0, 1]) Default: ``0.5``
        quantile_range ([float, float]): the quantile range of the registered images tracked during smoothing. Default: ``[0.1, 0.9]``
        sig_step (float): positive step size for smoothing sigmas. Default: ``0.5``
        use_gpu (bool): if ``True`` use gpu to smooth images. Default: ``False``

    Returns:
        minimal_sig (array_like): pixelwise template resolution measure (same shape as template)
    """

    nd = len(data_reg[0,].shape)
    minimal_sig = np.zeros_like(data_reg[0,])

    if use_gpu:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter as gaussian_filter_gpu
        data_reg_gpu = cp.asarray(data_reg)

    assert sig_step > 0, 'sig_step should be positive'
    q0, q1 = quantile_range
    assert (0.0 <= q0 <= 1.0) and (0.0 <= q1 <= 1.0) and (q0 < q1), 'quantile_range [q0, q1] entries q0 and q1 should be from the intervall [0, 1] and be ordered q0 < q1'

    mask_union = np.zeros_like(data_reg[0,], dtype=bool)

    sig = 0.0
    iter = 1
    while (~mask_union).any():
        
        if not use_gpu:
            reg_filter = gaussian_filter_cpu(data_reg, sigma=(0,) + nd * (sig,), mode='constant', cval=0.0)
            reg_filter_low, reg_filter_up = np.quantile(reg_filter, quantile_range, axis=0)
        
        if use_gpu:
            reg_filter_gpu = gaussian_filter_gpu(data_reg_gpu, sigma=(0,) + nd * (sig,), mode='constant', cval=0.0)
            
            # cupy.quantile/torch.quantile use too much memory for this dataset on a RTX 3090 ...
            # pytorch is faster at gpu to cpu transfer than cupy ...
            # torch.quantile on cpu uses multiple cores!
            qr = np.array(quantile_range).astype(reg_filter_gpu.dtype)
            reg_filter_low, reg_filter_up = t.cat([
                    t.quantile(chunk, q=t.tensor(qr), axis=0)
                    for chunk in t.chunk(t.as_tensor(reg_filter_gpu), chunks=10, dim=-1)
                ], dim=-1).numpy()
            # to save memory the dataset was split into chunks; quantiles are computed over all images for each chunk of pixels and then concatenated to complete images
            # version without chunks:
            #reg_filter_low, reg_filter_up = t.quantile(t.as_tensor(reg_filter_gpu), t.tensor(quantile_range), axis=0).numpy()

            # unused std version:
            #reg_filter_std = np.asnumpy(reg_filter.std(axis=0))

            del reg_filter_gpu
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        
        reg_filter_range = reg_filter_up - reg_filter_low
        #reg_filter_range = reg_filter_std

        mask_all = (reg_filter_range <= eff_height * (q1 - q0)) # all pixels that fulfill the quantile condition for the current sigma
        mask_new = mask_all & ~mask_union # only keep the pixels that have not fulfilled the condition before
        mask_union |= mask_all # union of masks that collects all pixels that already fulfilled the condition at least once
        
        minimal_sig[mask_new] = sig

        print('iter:', iter, 'sig:', sig, 'percent of pixels done:', 100 * mask_union.sum()//mask_union.size, end='\r')
        sig += sig_step
        iter += 1
    print('')
    
    return minimal_sig
