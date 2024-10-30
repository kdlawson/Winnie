from tqdm.auto import tqdm
from copy import copy, deepcopy
from winnie.convolution import (convolve_with_spatial_psfs, psf_convolve_cpu, psf_convolve_gpu)
from winnie.utils import crop_data
import numpy as np

def coronagraphic_richardson_lucy(image, psfs, psf_inds=None, im_mask=None, num_iter=500, im_deconv_in=None,
                                  epsilon=0, return_iters=None, use_gpu=False, ncores=-2, show_progress=True,
                                  excl_mask=None):
    float_type = image.dtype

    if im_deconv_in is None:
        im_deconv = np.full(image.shape, np.nanmedian(image[image>0]), dtype=float_type)
    else:
        im_deconv = im_deconv_in.copy()
        
    if np.ndim(psfs)==2:
        convolution_fn = psf_convolve_gpu if use_gpu else psf_convolve_cpu
        conv_kwargs = {}
    else:
        convolution_fn = convolve_with_spatial_psfs
        conv_kwargs = {'psf_inds':psf_inds, 'use_gpu':use_gpu, 'ncores':ncores}

    if im_mask is None:
        im_mask = 1.

    psfs_inv = np.flip(psfs, axis=(-1,-2))
    unity_conv = convolution_fn(np.ones_like(image), psfs_inv, **conv_kwargs) # Correction for shift-variant PSFs
    
    if return_iters is not None:
        im_dc_iters = np.zeros((len(return_iters), *image.shape), dtype=float_type)
        
    iterator = range(num_iter)
    if show_progress:
        iterator = tqdm(iterator, leave=False)
    for i in iterator:
        conv = convolution_fn(im_deconv*im_mask, psfs, **conv_kwargs)
        conv += np.sign(conv)*1e-12 # avoid division by very near zero values

        if epsilon != 0 and (i > 0 or im_deconv_in is not None):
            relative_blur = np.where(np.abs(conv) < epsilon, 0, image / conv)
        else:
            relative_blur = image / conv
        fcorr = convolution_fn(relative_blur, psfs_inv, **conv_kwargs)/unity_conv
        if excl_mask is not None:
            fcorr = np.where(excl_mask, 1, fcorr)
        im_deconv *= fcorr
        if return_iters is not None:
            if i in return_iters:
                im_dc_iters[return_iters == i] = im_deconv.copy()

    if return_iters is not None:
        return im_deconv, im_dc_iters
    return im_deconv