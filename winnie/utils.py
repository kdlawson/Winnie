import numpy as np
import astropy.units as u
from scipy import ndimage
from joblib import Parallel, delayed


def dist_to_pt(pt, nx=201, ny=201, dtype=np.float32):
    """
    Returns a square distance array of size (naxis,naxis), 
    where each pixel corresponds to the euclidean distance of that pixel from the center.
    """
    xaxis = np.arange(0, nx, dtype=dtype)-pt[0]
    yaxis = np.arange(0, ny, dtype=dtype)-pt[1]
    return np.sqrt(xaxis**2 + yaxis[:, np.newaxis]**2)


def propagate_nans_in_spatial_operation(a, fn, fn_args=None,
                                        fn_kwargs=None,
                                        fn_nan_kwargs=None,
                                        fn_zero_kwargs=None,
                                        prop_threshold=0,
                                        prop_zeros=True):
    """
    This takes an array, a, and and a function that performs some spatial operation on a, fn,
    and endeavours to propgate any nans (and optionally: zeros, which are often also non-physical values)
    through the indicated operation. Note: this operation is intentionally liberal with propgating the specified values.
    I.e., for rotation of an image with nans, expect there to be more NaN pixels following the operation. 
    This can be tuned somewhat by increasing the value of prop_threshold (0 <= prop_threshold <= 1)
    
    Example:
    import numpy as np
    from scipy import ndimage
    im = np.random.normal(loc=10, size=(101,101))
    im = ndimage.gaussian_filter(im, sigma=2.5)
    im[68:75, 34:48] = np.nan
    im[11:22, 8:19] = 0.
    angle = 30.0 # angle to rotate image by
    im_rot = propagate_nans_in_spatial_operation(im, ndimage.rotate, fn_args=[angle],
                                                 fn_kwargs=dict(axes=(-2, -1), reshape=False, cval=np.nan, prefilter=False),
                                                 fn_nan_kwargs=dict(axes=(-2, -1), reshape=False, prefilter=False),
                                                 prop_threshold=0, prop_zeros=True)
    """
    if fn_args is None: fn_args = []
    if fn_kwargs is None: fn_kwargs = {}
    if fn_nan_kwargs is None: fn_nan_kwargs = fn_kwargs
    
    nans = np.isnan(a)
    any_nans = np.any(nans)
    if any_nans:
        a_out = fn(np.where(nans, 0., a), *fn_args, **fn_kwargs)
    else: 
        a_out = fn(a, *fn_args, **fn_kwargs)
        
    if prop_zeros:
        zeros = a == 0.
        any_zeros = np.any(zeros)
        # Apply the operation to the boolean map of zeros 
        # >>> replace any locations > prop_threshold with zeros in the output
        if any_zeros:
            if fn_zero_kwargs is None:
                fn_zero_kwargs = fn_nan_kwargs
            zeros_out = fn(zeros.astype(float), *fn_args, **fn_zero_kwargs)
            a_out = np.where(zeros_out>prop_threshold, 0., a_out)
    if any_nans:
        nans_out = fn(nans.astype(float), *fn_args, **fn_nan_kwargs)
        a_out = np.where(nans_out>prop_threshold, np.nan, a_out)
    return a_out


def ang_size_to_proj_sep(ang_size, distance):
    """
    Converts angular separation (any angular unit, e.g. arcsec, degrees, radians) to projected separation (au).
    ang_size and distance can be provided as a float/int (in which case units of arcsec and parsec are assumed
    respectively).
    
    If not specified, units for ang_size and distance are assumed to be arcseconds and parsecs respectively.
    
    Example:
        1) r = ang_size_to_proj_sep(0.25, 156) 
        # Returns the proj separation in astropy units of au for an angular separation of 0.25 arcsec at 156 pc
        
        2) r = ang_size_to_proj_sep(250*u.mas, 508.8*u.lightyear) 
        # Returns roughly the same value as example 1, but with different input units.
        
    Note: returns an astropy unit value. 
          ang_size_to_proj_sep(ang_size, distance).value will give you a float instead.
    """
    ang_size = ang_size << u.arcsec # If units aren't provided, sets unit to arcsec. Else converts unit to arcsec
    d = distance << u.pc
    return (d * np.tan(ang_size.to('rad'))).to('AU')


def proj_sep_to_ang_size(proj_sep, distance):
    """
    Converts projected size (any unit of length) to angular separation (in arcsec)
    
    If not specified, units for proj_sep and distance are assumed to be AU and parsec respectively.
    """
    r = proj_sep << u.au
    d = distance << u.pc
    r = r << u.pc
    return np.arctan2(r, d).to(u.arcsec)


def ang_size_to_px_size(ang_size, pxscale):
    """
    Converts an angular separation (any angular unit) to pixels (based on pixel scale provided).
    
    If not specified, units for ang_size and pxscale are assumed to be arcseconds and arcsec/pixel respectively.
    """
    ang_size = ang_size << u.arcsec
    pxscale = pxscale << u.arcsec / u.pixel
    return ang_size / pxscale


def px_size_to_ang_size(px_size, pxscale):
    """
    Converts a pixel size (in pixels) to an angular separation (in arcsec).
    If not specified, units for px_size and pxscale are assumed to be pixels and arcsec/pixel respectively.
    """
    px_size = px_size << u.pixel
    pxscale = pxscale << u.arcsec / u.pixel
    return px_size * pxscale


def nan_median_absolute_deviation(x, axis=None, scaled=True, return_median=False, keepdims=False):
    """
    Median absolute deviation, optionally scaled to serve as a consistent estimator
    """
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=axis, keepdims=keepdims)
    if scaled:
        mad *= 1.4826
    if return_median:
        return (mad, med)
    return mad


def sigma_clipped_axis_nanmean(x, n=3., axis=0, clip_mask=None, return_clip_mask=False):
    """
    Computes nanmean of x along the indicated axis, clipping values further than n MAD from the median (non-iteratively).
    """
    if clip_mask is None:
        sig_mad, median = nan_median_absolute_deviation(x, axis=axis, scaled=True, return_median=True)
        clip_mask = np.logical_or(np.abs(x - median) > n * sig_mad, ~np.isfinite(x))
    
    xma = np.where(clip_mask, np.nan, x)
    x_out = np.nanmean(xma, axis=axis)
    if return_clip_mask:
        return x_out, clip_mask
    return x_out


def robust_mean_combine(imcube, errcube=None, robust_clip_nsig=3):
    im, clip_mask = sigma_clipped_axis_nanmean(imcube, n=robust_clip_nsig, axis=0, return_clip_mask=True)
    if errcube is None:
        return im,None
    n = np.sum(~clip_mask, axis=0)
    err = np.sqrt(np.nansum(np.where(clip_mask, np.nan, errcube**2), axis=0))/n
    return im,err
    
    
def median_combine(imcube, errcube=None):
    im = np.nanmedian(imcube, axis=0)
    if errcube is None:
        return im,None 
    n = np.sum(np.isfinite(imcube), axis=0)
    sig_mean = np.sqrt(np.nansum(errcube**2, axis=0))/n
    err = np.sqrt(np.pi*(2*n+1)/(4*n)) * sig_mean
    return im,err


def gaussian_filter_sequence(im, sigma, prop_threshold=1e-6):
    im = np.asarray(im)
    nd = np.ndim(im)
    fn_args = []
    fn_kwargs = dict(sigma=sigma)
    if nd == 2:
        im_out = propagate_nans_in_spatial_operation(im, ndimage.gaussian_filter, fn_args=fn_args,
                                                     fn_kwargs=fn_kwargs, prop_threshold=prop_threshold)
    else:
        ny, nx = im.shape[-2:]
        nI = np.prod(im.shape[:-2])
        im_reshaped = im.reshape((nI, ny, nx))
        im_out = np.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_out[i] = propagate_nans_in_spatial_operation(im_reshaped[i], ndimage.gaussian_filter,
                                                            fn_args=fn_args, fn_kwargs=fn_kwargs, prop_threshold=prop_threshold)
        im_out = im_out.reshape(im.shape)
    return im_out


def high_pass_filter_sequence(im, filtersize, prop_threshold=1e-6):
    im = np.asarray(im)
    return im-gaussian_filter_sequence(im, filtersize, prop_threshold=prop_threshold)


def median_filter_sequence(im, radius=None, size=None, footprint=None, prop_threshold=1e-6):
    im = np.asarray(im)
    nd = np.ndim(im)
    fn_args = []
    fn_kwargs = dict(size=size, footprint=footprint)
    if radius is not None and footprint is None:
        rnx = rny = int(np.ceil(radius)*2+1)
        rc0 = np.array([(rnx-1)/2.,(rnx-1)/2.])
        footprint = dist_to_pt(rc0, nx=rnx, ny=rny) <= radius
        fn_kwargs['footprint'] = footprint
    if nd == 2:
        im_out = propagate_nans_in_spatial_operation(im, ndimage.median_filter, fn_args=fn_args, fn_kwargs=fn_kwargs, prop_threshold=prop_threshold)
    else:
        ny, nx = im.shape[-2:]
        nI = np.prod(im.shape[:-2])
        im_reshaped = im.reshape((nI, ny, nx))
        im_out = np.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_out[i] = propagate_nans_in_spatial_operation(im_reshaped[i], ndimage.median_filter, fn_args=fn_args, fn_kwargs=fn_kwargs, prop_threshold=prop_threshold)
        im_out = im_out.reshape(im.shape)
    return im_out

def pad_or_crop_image(im, new_size, cent=None, new_cent=None, cval0=np.nan, nan_prop_threshold=0., zero_prop_threshold=0., prefilter=True, order=3):
    new_size = np.asarray(new_size)
    im_size = np.array(im.shape)
    ny, nx = im_size
    if cent is None:
        cent = (np.array([nx,ny])-1.)/2.
        
    if new_cent is None:
        new_cent = (np.array([new_size[1],new_size[0]])-1.)/2.
        
    if np.all([new_size == im_size, cent == new_cent]):
        return im.copy()
    
    if np.all([float(i).is_integer() for i in [*cent, *new_cent]]):
        # No need to treat nans/zeros differently if both centers are integers.
        out_im = pad_or_crop_about_pos(im, cent, new_size, new_cent=new_cent, cval=cval0, prefilter=False, order=0)
    else:    
        nans = np.isnan(im)
        zeros = im == 0.
        any_zeros = np.any(zeros)
        any_nans = np.any(nans)
        if any_nans:
            out_im = pad_or_crop_about_pos(np.where(nans, 0., im), cent, new_size, new_cent=new_cent, cval=cval0, prefilter=prefilter, order=order)
        else:
            out_im = pad_or_crop_about_pos(im, cent, new_size, new_cent=new_cent, cval=cval0, prefilter=prefilter, order=order)
        if any_zeros:
            out_zeros = pad_or_crop_about_pos(zeros.astype(float), cent, new_size, new_cent=new_cent, prefilter=False)
            out_im = np.where(out_zeros>zero_prop_threshold, 0., out_im)
        if any_nans:
            out_nans = pad_or_crop_about_pos(nans.astype(float), cent, new_size, new_cent=new_cent, prefilter=False)
            out_im = np.where(out_nans>nan_prop_threshold, np.nan, out_im)
    return out_im


def pad_or_crop_about_pos(im, pos, new_size, new_cent=None, cval=np.nan, order=3, mode='constant', prefilter=True):
    ny, nx = im.shape[-2:]
    ny_new, nx_new = new_size
    if new_cent is None:
        new_cent = (np.array([nx_new,ny_new])-1.)/2.
        
    nd = np.ndim(im)
    xg, yg = np.meshgrid(np.arange(nx_new, dtype=np.float32), np.arange(ny_new, dtype=np.float32))
    
    xg -= (new_cent[0]-pos[0])
    yg -= (new_cent[1]-pos[1])
    if nd == 2:
        im_out = ndimage.map_coordinates(im, np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
    else:
        nI = np.product(im.shape[:-2])
        im_reshaped = im.reshape((nI, ny, nx))
        im_out = np.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_out[i] = ndimage.map_coordinates(im_reshaped[i], np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
        im_out = im_out.reshape((*im.shape[:-2], ny, nx))
    return im_out


def crop_data(data, cent, new_shape, return_indices=False, copy=True):
    new_ny, new_nx = new_shape
    x0, y0 = cent
    x1, y1 = max(0, int(np.round(x0-(new_nx-1.)/2.))), max(0, int(np.round(y0-(new_ny-1.)/2.)))
    x2, y2 = x1+new_nx, y1+new_ny
    data_cropped = (data[..., y1:y2, x1:x2].copy() if copy else data[..., y1:y2, x1:x2])
    new_cent = np.array([x0-x1, y0-y1])
    if return_indices:
        return data_cropped, new_cent, [y1,y2,x1,x2]
    return data_cropped, new_cent


def xy_polar_ang_displacement(x, y, dtheta):
    """
    Rotates cartesian coordinates x and y by angle dtheta (deg) about (0,0).
    """
    r = np.sqrt(x**2+y**2)
    theta = np.rad2deg(np.arctan2(y,x))
    new_theta = np.deg2rad(theta+dtheta)
    newx,newy = r*np.cos(new_theta),r*np.sin(new_theta)
    return newx,newy


def c_to_c_osamp(c, osamp):
    return np.asarray(c)*osamp + 0.5*(osamp-1)


def xy_polar_ang_displacement_gpu(x, y, dtheta):
    r = cp.sqrt(x**2+y**2)
    theta = cp.rad2deg(cp.arctan2(y,x))
    new_theta = cp.deg2rad(theta+dtheta)
    newx,newy = r*cp.cos(new_theta),r*cp.sin(new_theta)
    return newx,newy


def rotate_image(im, angle, cent=None, new_cent=None, cval0=np.nan, prefilter=True, use_gpu=False):
    if use_gpu:
        return rotate_image_gpu(im, angle, cent=cent, new_cent=new_cent, cval0=cval0, prefilter=prefilter)
    return rotate_image_cpu(im, angle, cent=cent, new_cent=new_cent, cval0=cval0, prefilter=prefilter)


def rotate_image_cpu(im, angle, cent=None, new_cent=None, cval0=np.nan, prefilter=True):
    """
    Rotates im by angle "angle" in degrees using CPU operations. Avoids "mixing" exact zero values,
    which should functionally be treated as nans. If cent is provided, rotates about cent. 
    Otherwise, uses ndimage's rotate (which is a bit faster) to rotate about the geometric center.
    """

    if angle == 0.:
        return im.copy()
    nans = np.isnan(im)
    zeros = im == 0.
    any_zeros = np.any(zeros)
    any_nans = np.any(nans)
    if cent is None:
        if any_nans:
            rot_im = ndimage.rotate(np.where(nans, 0., im), angle, axes=(-2, -1), reshape=False, cval=cval0, prefilter=prefilter)
        else:
            rot_im = ndimage.rotate(im, angle, axes=(-2, -1), reshape=False, cval=cval0, prefilter=prefilter)
        if any_zeros:
            rot_zeros = ndimage.rotate(zeros.astype(float), angle, axes=(-2, -1),  prefilter=False, reshape=False)
            rot_im = np.where(rot_zeros>0., 0., rot_im)
        if any_nans:
            rot_nans = ndimage.rotate(nans.astype(float), angle, axes=(-2, -1),  prefilter=False, reshape=False)
            rot_im = np.where(rot_nans>0., np.nan, rot_im)
    else:
        if any_nans:
            rot_im = rotate_about_pos(np.where(nans, 0., im), cent, angle, new_cent=new_cent, cval=cval0, prefilter=prefilter)
        else:
            rot_im = rotate_about_pos(im, cent, angle, new_cent=new_cent, cval=cval0, prefilter=prefilter)
        if any_zeros:
            rot_zeros = rotate_about_pos(zeros.astype(float), cent, angle, new_cent=new_cent, prefilter=False)
            rot_im = np.where(rot_zeros>0., 0., rot_im)
        if any_nans:
            rot_nans = rotate_about_pos(nans.astype(float), cent, angle, new_cent=new_cent, prefilter=False)
            rot_im = np.where(rot_nans>0., np.nan, rot_im)
    return rot_im


def rotate_image_gpu(im0, angle, cent=None, cval0=np.nan, prefilter=True):
    """
    Rotates im by angle "angle" in degrees using GPU operations. Avoids "mixing" exact zero values, which should functionally be treated as nans.
    If cent is provided, rotates about cent. Otherwise, uses CuPy's version of scipy.ndimage's rotate (which is a bit faster) to rotate about the
    geometric center.
    """
    if angle == 0.:
        return im0.copy()
    im = cp.asarray(im0)
    nans = cp.isnan(im)
    zeros = im == 0.
    any_zeros = cp.any(zeros)
    any_nans = cp.any(nans)
    if cent is None:
        if any_nans:
            rot_im = cp_ndimage.rotate(cp.where(nans, 0., im), angle, axes=(-2, -1), reshape=False, cval=cval0, prefilter=prefilter)
        else:
            rot_im = cp_ndimage.rotate(im, angle, axes=(-2, -1), reshape=False, cval=cval0, prefilter=prefilter)
        if any_zeros:
            rot_zeros = cp_ndimage.rotate(zeros.astype(float), angle, axes=(-2, -1),  prefilter=False, reshape=False)
            rot_im = cp.where(rot_zeros>0., 0., rot_im)
        if any_nans:
            rot_nans = cp_ndimage.rotate(nans.astype(float), angle, axes=(-2, -1),  prefilter=False, reshape=False)
            rot_im = cp.where(rot_nans>0., cp.nan, rot_im)
    else:
        if any_nans:
            rot_im = rotate_about_pos_gpu(cp.where(nans, 0., im), cent, angle, cval=cval0, prefilter=prefilter)
        else:
            rot_im = rotate_about_pos_gpu(im, cent, angle, cval=cval0, prefilter=prefilter)
        if any_zeros:
            rot_zeros = rotate_about_pos_gpu(zeros.astype(float), cent, angle,  prefilter=False)
            rot_im = cp.where(rot_zeros>0., 0., rot_im)
        if any_nans:
            rot_nans = rotate_about_pos_gpu(nans.astype(float), cent, angle,  prefilter=False)
            rot_im = cp.where(rot_nans>0., cp.nan, rot_im)
    return cp.asnumpy(rot_im)


def rotate_about_pos(im, pos, angle, new_cent=None, cval=np.nan, order=3, mode='constant', prefilter=True):
    ny, nx = im.shape[-2:]
    nd = np.ndim(im)
    yg0, xg0 = np.indices((ny,nx), dtype=np.float64)
    
    if new_cent is not None:
        xg0 -= (new_cent[0]-pos[0])
        yg0 -= (new_cent[1]-pos[1])
    
    xg,yg = xy_polar_ang_displacement(xg0-pos[0], yg0-pos[1], angle)
    xg += pos[0]
    yg += pos[1]
    if nd == 2:
        im_rot = ndimage.map_coordinates(im, np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
    else:
        nI = np.prod(im.shape[:-2])
        im_reshaped = im.reshape((nI, ny, nx))
        im_rot = np.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_rot[i] = ndimage.map_coordinates(im_reshaped[i], np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
        im_rot = im_rot.reshape((*im.shape[:-2], ny, nx))
    return im_rot


def rotate_about_pos_gpu(im, pos, angle, cval=np.nan, order=3, mode='constant', prefilter=True):
    ny, nx = im.shape[-2:]
    nd = cp.ndim(im)
    xg0, yg0 = cp.meshgrid(cp.arange(nx, dtype=cp.float64), cp.arange(ny, dtype=cp.float64))
    
    xg,yg = xy_polar_ang_displacement_gpu(xg0-pos[0], yg0-pos[1], angle)
    xg += pos[0]
    yg += pos[1]
    
    if nd == 2:
        im_rot = cp_ndimage.map_coordinates(im, cp.array([yg,xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
    else:
        nI = int(cp.prod(cp.array(im.shape[:-2])))
        im_reshaped = im.reshape((nI, ny, nx))
        im_rot = cp.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_rot[i] = cp_ndimage.map_coordinates(im_reshaped[i], cp.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
        im_rot = im_rot.reshape((*im.shape[:-2], ny, nx))
    xg, yg, xg0, yg0 = free_gpu(xg, yg, xg0, yg0)
    return im_rot


def rotate_hypercube(hcube, angles, cent=None, new_cent=None, ncores=-1, use_gpu=False, cval0=0., prefilter=True):
    """
    Rotates an N-dimensional array, 'hcube', where the final two axes are assumed to be cartesian y and x 
    and where 'angles' is an array of angles (in degrees) matching the length of the first dimension.
    
    E.g., for a sequence of nT images having shape (ny,nx), hcube should have shape (nT,ny,nx) and angles should have shape (nT,)
    
    For a sequence of nT IFS image cubes each having nL wavelength images of shape (ny,nx), hcube should have shape (nT, nL, ny, nx)
    """
    if use_gpu:
        rot_hcube = np.stack([rotate_image_gpu(imcube, angle, cval0=cval0, cent=cent, new_cent=new_cent, prefilter=prefilter) for imcube, angle in zip(hcube, angles)])
    else:
        rot_hcube = np.stack(Parallel(n_jobs=ncores, prefer='threads')(delayed(rotate_image_cpu)(imcube, angle, cval0=cval0, cent=cent, new_cent=new_cent, prefilter=prefilter) for imcube, angle in zip(hcube, angles)))
    return rot_hcube


def pad_and_rotate_hypercube(hcube, angles, cent=None, ncores=-1, use_gpu=False, cval0=np.nan, prefilter=True):
    ny, nx = hcube.shape[-2:]
    if cent is None:
        cent = (np.array([nx, ny])-1.)/2.
    dymin_pad, dymax_pad, dxmin_pad, dxmax_pad = compute_derot_padding(nx, ny, angles, cent=cent)
    hcube_pad = np.pad(hcube.copy(), [*[[0,0] for _ in range(hcube.ndim-2)], [dymin_pad, dymax_pad], [dxmin_pad, dxmax_pad]], constant_values=np.nan)
    cent_pad = cent + np.array([dxmin_pad, dymin_pad])
    hcube_pad_rot = rotate_hypercube(hcube_pad, angles, cent=cent_pad, ncores=ncores, use_gpu=use_gpu, cval0=cval0, prefilter=prefilter)
    return hcube_pad_rot, cent_pad

def compute_derot_padding(nx, ny, angles, cent=None):
    if cent is None:
        cent = (np.array([nx, ny])-1.)/2.
    dxmin, dxmax = np.array([0, nx]) - cent[0]
    dymin, dymax = np.array([0, ny]) - cent[1]
    corner_coords = np.array([[dxmax, dymax],
                              [dxmax, dymin],
                              [dxmin, dymin],
                              [dxmin, dymax]])
    uni_angs = np.unique(angles)
    derot_corner_coords = np.vstack([np.array(xy_polar_ang_displacement(*corner_coords.T, -ang)).T for ang in uni_angs])
    dxmin_pad, dymin_pad = (np.ceil(np.abs(np.min(derot_corner_coords, axis=0) - np.array([dxmin, dymin])))).astype(int)
    dxmax_pad, dymax_pad = (np.ceil(np.abs(np.max(derot_corner_coords, axis=0) - np.array([dxmax, dymax])))).astype(int)
    return dymin_pad, dymax_pad, dxmin_pad, dxmax_pad

def free_gpu(*args):
    N = len(args)
    args = list(args)
    for i in range(N):
        args[i] = None
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    if N <= 1:
        return None
    return args


try:
    import cupy as cp # type: ignore
    from cupyx.scipy import ndimage as cp_ndimage # type: ignore
    gpu = cp.cuda.Device(0)
except ModuleNotFoundError:
    pass