from tqdm.auto import tqdm
from copy import copy, deepcopy
from winnie.convolution import (SpatialPSFConvolver, _get_psf_patch_bounds)
from winnie.backend import (get_device, resolve_dtype, to_tensor, to_numpy)
from winnie.utils import crop_data
import numpy as np
import torch

def coronagraphic_richardson_lucy(image, psfs, psf_inds=None, im_mask=None, num_iter=500, im_deconv_in=None,
                                  epsilon=0, return_iters=None, device=None, show_progress=True,
                                  excl_mask=None, mirror_psfs=False, convolution_method='auto', 
                                  conv_patch_bounds=None, conv_used_inds=None):
    """
    Richardson-Lucy deconvolution of image for coronagraphic data, where the PSF varies spatially (if psfs is a 3D stack,
    in which case psf_inds indicates which PSF to use for each pixel) and the transmission of the coronagraph is given by
    im_mask. The calculations are carried out on device (see winnie.backend) and are performed in float32 if the device
    does not support float64 (e.g., 'mps'). Returns the deconvolved image as a numpy array with the same dtype as image
    (and the images at the iterations in return_iters, if provided).
    """
    float_type = image.dtype
    device = get_device(device)
    dtype = resolve_dtype(device, image.dtype, psfs.dtype)

    if im_deconv_in is None:
        im_deconv = np.full(image.shape, np.nanmedian(image[image>0]), dtype=float_type)
    else:
        im_deconv = im_deconv_in
        
    psfs_inv = np.flip(psfs, axis=(-1,-2)) if mirror_psfs else psfs
    convolve = SpatialPSFConvolver(psfs, device=device)
    convolve_inv = SpatialPSFConvolver(psfs_inv, device=device) if mirror_psfs else convolve

    if np.ndim(psfs)==2:
        conv_kwargs = {}
    else:
        if (conv_used_inds is None) or (conv_patch_bounds is None):
            psf_yhw, psf_xhw = np.ceil(np.array(psfs.shape[-2:])/2.).astype(int)
            conv_used_inds, conv_patch_bounds = _get_psf_patch_bounds(psf_inds, psfs.shape[0], (psf_yhw, psf_xhw))
        conv_kwargs = {'psf_inds':to_tensor(psf_inds, device), 'method':convolution_method, 
                       'patch_bounds':conv_patch_bounds, 'used_inds':conv_used_inds}

    image_t = to_tensor(image, device, dtype)
    im_deconv_t = to_tensor(im_deconv, device, dtype).clone()
    im_mask_t = 1. if im_mask is None else to_tensor(im_mask, device, dtype)
    excl_mask_t = None if excl_mask is None else to_tensor(excl_mask, device, torch.bool)

    unity_conv = convolve_inv(torch.ones_like(image_t), **conv_kwargs) # Correction for shift-variant PSFs
    
    if return_iters is not None:
        im_dc_iters = np.zeros((len(return_iters), *image.shape), dtype=float_type)
        
    iterator = range(num_iter)
    if show_progress:
        iterator = tqdm(iterator, leave=False)
    for i in iterator:
        conv = convolve(im_deconv_t*im_mask_t, **conv_kwargs)
        conv += torch.sign(conv)*1e-12 # avoid division by very near zero values

        relative_blur = image_t / conv
        if epsilon != 0 and (i > 0 or im_deconv_in is not None):
            relative_blur = relative_blur.masked_fill(torch.abs(conv) < float(epsilon), 0.)
        fcorr = convolve_inv(relative_blur, **conv_kwargs)/unity_conv
        if excl_mask_t is not None:
            fcorr = fcorr.masked_fill(excl_mask_t, 1.)
        im_deconv_t *= fcorr
        if return_iters is not None:
            if i in return_iters:
                im_dc_iters[return_iters == i] = to_numpy(im_deconv_t, float_type)

    im_deconv = to_numpy(im_deconv_t, float_type)
    if return_iters is not None:
        return im_deconv, im_dc_iters
    return im_deconv