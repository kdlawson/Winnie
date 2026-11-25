import numpy as np
import astropy.units as u
import torch
import torch.nn.functional as F
from scipy import ndimage
from photutils.aperture import EllipticalAnnulus, aperture_photometry
from .backend import get_device, resolve_dtype, to_tensor, to_numpy, chunk_ranges


def dist_to_pt(pt, nx=201, ny=201, dtype=np.float32, return_xy=False):
    """
    Returns a square distance array of size (naxis,naxis), 
    where each pixel corresponds to the euclidean distance of that pixel from the center.
    """
    xaxis = np.arange(0, nx, dtype=dtype)-pt[0]
    yaxis = np.arange(0, ny, dtype=dtype)-pt[1]
    rg = np.sqrt(xaxis**2 + yaxis[:, np.newaxis]**2)
    if return_xy:
        xg = np.broadcast_to(xaxis, rg.shape)
        yg = np.broadcast_to(yaxis[:, np.newaxis], rg.shape)
        return (xg, yg, rg)
    return rg


def propagate_nans_in_spatial_operation(a, fn, fn_args=None,
                                        fn_kwargs=None,
                                        fn_nan_kwargs=None,
                                        fn_zero_kwargs=None,
                                        prop_threshold=0,
                                        prop_zeros=True):
    """
    This takes an array, a, and and a function that performs some spatial operation on a, fn,
    and endeavours to propagate any nans (and optionally: zeros, which are often also non-physical values)
    through the indicated operation. Note: this operation is intentionally liberal with propagating the specified values.
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
    
    nans = ~np.isfinite(a)
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
    medsig_sfac = np.where(n<=2, 1, np.sqrt(np.pi*(2*n+1)/(4*n)))
    err = medsig_sfac * sig_mean
    return im,err


def mean_combine(imcube, errcube=None):
    im = np.nanmean(imcube, axis=0)
    if errcube is None:
        return im,None 
    n = np.sum(np.isfinite(imcube), axis=0)
    err = np.sqrt(np.nansum(errcube**2, axis=0))/n
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


def high_pass_filter_sequence(im, filtersize, prop_threshold=1e-6, niter=1):
    im = np.asarray(im)
    im_out = im.copy()
    for _ in range(niter):
        im_out -= gaussian_filter_sequence(im_out, filtersize, prop_threshold=prop_threshold)
    return im_out


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

# Interpolation orders available through torch's grid_sample. Higher order splines (e.g., order=5 when shifting PSFs) are
# computed with scipy on the CPU instead.
_INTERP_MODES = {0: 'nearest', 1: 'bilinear', 3: 'bicubic'}
_MAX_CHUNK_BYTES = 2**30


def _spline_interp(t, x, y, order, cval):
    a = to_numpy(t)
    coords = np.array([to_numpy(y[0]), to_numpy(x[0])], dtype=np.float64)
    out = np.stack([ndimage.map_coordinates(im_i, coords, order=order, mode='constant', cval=cval)
                    for im_i in a.reshape((-1, *a.shape[-2:]))])
    return to_tensor(out.reshape((*a.shape[:-2], *coords.shape[1:])), t.device, t.dtype)


def _grid_interp(t, x, y, order, cval):
    """
    Interpolates the tensor t, with shape (B, C, ny, nx), at the pixel coordinates (x, y) using torch's grid_sample. The
    coordinates must broadcast to shape (B, ny_out, nx_out) (or to (1, ny_out, nx_out) if they are shared by all B
    slices of t), with pixel centers at integer values. Samples outside the image (beyond the centers of the edge
    pixels) are set to cval.
    """
    B, C, ny, nx = t.shape
    if min(ny, nx) < 2:
        raise ValueError("Interpolation requires at least two pixels along each axis.")
    x, y = torch.broadcast_tensors(x, y)
    if x.ndim == 2:
        x, y = x[None], y[None]
    if order not in _INTERP_MODES:
        return _spline_interp(t, x, y, order, cval)
    grid = torch.stack((x * (2. / (nx - 1)) - 1., y * (2. / (ny - 1)) - 1.), dim=-1)
    if grid.shape[0] != B:
        grid = grid.expand(B, -1, -1, -1).contiguous()
    # Reflecting at the edges is analogous to the boundary conditions of spline interpolation, so pixels near the edges
    # are not pulled toward zero. Anything actually outside of the image is replaced with cval below.
    out = F.grid_sample(t, grid, mode=_INTERP_MODES[order], padding_mode='reflection', align_corners=True)
    tol = 1e-3 if t.dtype == torch.float32 else 1e-9 # guards against floating point rounding of the coordinates
    outside = (x < -tol) | (x > nx - 1 + tol) | (y < -tol) | (y > ny - 1 + tol)
    return out.masked_fill(outside[:, None], cval)


def _interp_propagating(t, x, y, order, cval, zero_threshold, nan_threshold, dilate):
    """
    Interpolates t like _grid_interp, but treats non-finite values as missing and exact zeros as invalid: any output
    pixel whose interpolated zero-mask exceeds zero_threshold is set to exactly zero, and any whose interpolated
    non-finite-mask exceeds nan_threshold is set to NaN. If dilate is True, the masks are dilated by one pixel before
    interpolation so that they cover the full footprint of a bicubic kernel (rather than that of a bilinear one).
    """
    B, C, ny, nx = t.shape
    nans, zeros = ~torch.isfinite(t), t == 0
    any_nans, any_zeros = bool(nans.any()), bool(zeros.any())
    out = _grid_interp(t.masked_fill(nans, 0.) if any_nans else t, x, y, order, cval)
    flags = [m for m, any_m in ((zeros, any_zeros), (nans, any_nans)) if any_m]
    if flags:
        masks = torch.cat(flags, dim=1).to(t.dtype)
        if dilate:
            masks = F.max_pool2d(masks.reshape(-1, 1, ny, nx), 3, stride=1, padding=1).reshape(masks.shape)
        masks = _grid_interp(masks, x, y, 1, 0.)
        if any_zeros:
            out = out.masked_fill(masks[:, :C] > zero_threshold, 0.)
        if any_nans:
            out = out.masked_fill(masks[:, -C:] > nan_threshold, float('nan'))
    return out


def _out_dtype(im):
    return im.dtype if im.dtype.kind == 'f' else np.float64


def sample_image(im, x, y, order=3, cval=np.nan, device=None):
    """
    Interpolates im at the pixel coordinates (x, y) with torch's grid_sample, using the center of the first pixel as the
    origin.

    Parameters
    ----------
    im: ndarray
        Array of shape (..., ny, nx) to interpolate. The same coordinates are used for all leading axes.

    x, y: ndarray
        Arrays of shape (ny_out, nx_out) giving the x and y pixel coordinates (in the pixel frame of im) at which to
        sample.

    order: {0, 1, 3}
        Interpolation order: 0 (nearest), 1 (bilinear) or 3 (bicubic). Other orders are handled with scipy's spline
        interpolation on the CPU.

    cval: float
        Value assigned to samples that lie outside of the image.

    device: str or torch.device or None
        Device on which to carry out the interpolation. If None, the default device is used (see winnie.backend).

    Returns
    -------
    ndarray with shape (..., ny_out, nx_out).
    """
    im = np.asarray(im)
    device = get_device('cpu' if order not in _INTERP_MODES else device)
    dtype = resolve_dtype(device, im.dtype)
    t = to_tensor(im, device, dtype).reshape((1, -1, *im.shape[-2:]))
    out = _grid_interp(t, to_tensor(x, device, dtype), to_tensor(y, device, dtype), order, cval)
    return to_numpy(out, _out_dtype(im)).reshape((*im.shape[:-2], *out.shape[-2:]))


def pad_or_crop_image(im, new_size, cent=None, new_cent=None, cval0=np.nan, nan_prop_threshold=0., zero_prop_threshold=0., order=3, device=None):
    """
    Pads or crops the 2D image im to the shape new_size, placing the position cent of im (as [x, y]) at the position
    new_cent of the output (defaulting to the geometric centers). If the centers are integers, pixels are simply
    moved. Otherwise, the image is interpolated (see sample_image for the available interpolation orders).

    In the latter case, nans and exact zeros in the input (the latter being treated as invalid values) are propagated
    liberally: any output pixel for which the footprint of the interpolation kernel includes a nan (a zero) is set
    to nan (zero) if the interpolated indicator for it exceeds nan_prop_threshold (zero_prop_threshold).
    """
    new_size = np.asarray(new_size)
    im_size = np.array(im.shape)
    ny, nx = im_size
    cent = (np.array([nx, ny])-1.)/2. if cent is None else np.asarray(cent, dtype=float)
    new_cent = (np.array([new_size[1], new_size[0]])-1.)/2. if new_cent is None else np.asarray(new_cent, dtype=float)

    if np.array_equal(new_size, im_size) and np.array_equal(cent, new_cent):
        return im.copy()

    if np.all([float(i).is_integer() for i in [*cent, *new_cent]]):
        # No need to treat nans/zeros differently if both centers are integers.
        return pad_or_crop_about_pos(im, cent, new_size, new_cent=new_cent, cval=cval0, order=0, device=device)

    im = np.asarray(im)
    device = get_device('cpu' if order not in _INTERP_MODES else device)
    dtype = resolve_dtype(device, im.dtype)
    ny_new, nx_new = [int(n) for n in new_size]
    x = torch.arange(nx_new, dtype=dtype, device=device)[None, None, :] + float(cent[0] - new_cent[0])
    y = torch.arange(ny_new, dtype=dtype, device=device)[None, :, None] + float(cent[1] - new_cent[1])
    t = to_tensor(im, device, dtype)[None, None]
    out = _interp_propagating(t, x, y, order, cval0, zero_prop_threshold, nan_prop_threshold, dilate=True)
    return to_numpy(out[0, 0], _out_dtype(im))


def pad_or_crop_about_pos(im, pos, new_size, new_cent=None, cval=np.nan, order=3, device=None):
    """
    Pads or crops im (with shape (..., ny, nx)) to new_size such that the position pos (as [x, y]) of im is placed at
    new_cent in the output, with values outside of the input set to cval. No special treatment of nans or zeros is
    applied (see pad_or_crop_image for that).
    """
    im = np.asarray(im)
    ny_new, nx_new = [int(n) for n in new_size]
    if new_cent is None:
        new_cent = (np.array([nx_new, ny_new])-1.)/2.
    shift_x, shift_y = pos[0] - new_cent[0], pos[1] - new_cent[1] # output pixel (j, i) corresponds to input (j+shift_x, i+shift_y)

    if order == 0 and float(shift_x).is_integer() and float(shift_y).is_integer():
        # This is a pure shift, so we can simply copy the overlapping region
        shift_x, shift_y = int(shift_x), int(shift_y)
        ny, nx = im.shape[-2:]
        out = np.full((*im.shape[:-2], ny_new, nx_new), cval, dtype=_out_dtype(im))
        y1, y2 = max(0, -shift_y), min(ny_new, ny-shift_y)
        x1, x2 = max(0, -shift_x), min(nx_new, nx-shift_x)
        if (y2 > y1) and (x2 > x1):
            out[..., y1:y2, x1:x2] = im[..., y1+shift_y:y2+shift_y, x1+shift_x:x2+shift_x]
        return out

    x, y = np.meshgrid(np.arange(nx_new)+shift_x, np.arange(ny_new)+shift_y)
    return sample_image(im, x, y, order=order, cval=cval, device=device)


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


def xy_polar_ang_displacement(x, y, dtheta, x0=0, y0=0):
    """
    Rotates cartesian coordinates x and y by angle dtheta (deg) about [x0,y0].
    """
    dx, dy = x-x0, y-y0
    r = np.sqrt(dx**2+dy**2)
    theta = np.rad2deg(np.arctan2(dy, dx))
    new_theta = np.deg2rad(theta+dtheta)
    newx,newy = r*np.cos(new_theta)+x0, r*np.sin(new_theta)+y0
    return newx,newy


def c_to_c_osamp(c, osamp):
    return np.asarray(c)*osamp + 0.5*(osamp-1)


def _rotation_sampling_coords(shape, angles, cent, new_cent, dtype, device):
    """
    Returns the pixel coordinates (x, y), each with shape (len(angles), ny, nx), at which to sample an image in order to
    rotate it by each angle (in degrees) about cent (as [x, y]), with the result centered on new_cent (defaulting to
    cent). Rotations follow the convention of xy_polar_ang_displacement.
    """
    ny, nx = shape
    c_out = cent if new_cent is None else new_cent
    dx = (torch.arange(nx, dtype=dtype, device=device) - float(c_out[0]))[None, None, :]
    dy = (torch.arange(ny, dtype=dtype, device=device) - float(c_out[1]))[None, :, None]
    ang = np.deg2rad(np.asarray(angles, dtype=np.float64))
    cos_a = to_tensor(np.cos(ang), device, dtype)[:, None, None]
    sin_a = to_tensor(np.sin(ang), device, dtype)[:, None, None]
    return dx*cos_a - dy*sin_a + float(cent[0]), dx*sin_a + dy*cos_a + float(cent[1])


def rotate_image(im, angle, cent=None, new_cent=None, cval0=np.nan, device=None):
    """
    Rotates im (with shape (..., ny, nx)) by angle (in degrees) about cent (as [x, y]; defaults to the geometric center)
    using bicubic interpolation. Locations that fall outside of im are set to cval0. See rotate_hypercube for details
    on the treatment of nans and zeros.
    """
    im = np.asarray(im)
    if angle == 0.:
        return im.copy()
    return rotate_hypercube(im[np.newaxis], [angle], cent=cent, new_cent=new_cent, device=device, cval0=cval0)[0]


def rotate_about_pos(im, pos, angle, new_cent=None, cval=np.nan, order=3, device=None):
    """
    Rotates im (with shape (..., ny, nx)) by angle (in degrees) about pos (as [x, y]), placing the result at new_cent
    if provided. Unlike rotate_image, no special treatment is given to nans or zeros.
    """
    im = np.asarray(im)
    ny, nx = im.shape[-2:]
    device = get_device('cpu' if order not in _INTERP_MODES else device)
    dtype = resolve_dtype(device, im.dtype)
    x, y = _rotation_sampling_coords((ny, nx), [angle], pos, new_cent, dtype, device)
    out = _grid_interp(to_tensor(im, device, dtype).reshape((1, -1, ny, nx)), x, y, order, cval)
    return to_numpy(out, _out_dtype(im)).reshape(im.shape)


def rotate_hypercube(hcube, angles, cent=None, new_cent=None, device=None, cval0=0.):
    """
    Rotates an N-dimensional array, 'hcube', where the final two axes are assumed to be cartesian y and x 
    and where 'angles' is an array of angles (in degrees) matching the length of the first dimension. Rotation is about
    cent (as [x, y]; defaults to the geometric center) using bicubic interpolation, and locations that fall outside of
    the input are set to cval0.
    
    E.g., for a sequence of nT images having shape (ny,nx), hcube should have shape (nT,ny,nx) and angles should have shape (nT,)
    
    For a sequence of nT IFS image cubes each having nL wavelength images of shape (ny,nx), hcube should have shape (nT, nL, ny, nx)

    Non-finite values are treated as missing data and are propagated to any pixels whose interpolation involves them.
    Exact zeros, which are also treated as invalid values, are not "mixed" into neighboring pixels: any pixel that is 
    more than half composed of zeros after rotation is set to exactly zero.
    """
    hcube = np.asarray(hcube)
    nT, (ny, nx) = len(hcube), hcube.shape[-2:]
    angles = np.broadcast_to(np.asarray(angles, dtype=np.float64), (nT,))
    if cent is None:
        cent = (np.array([nx, ny])-1.)/2.
    device = get_device(device)
    dtype = resolve_dtype(device, hcube.dtype)
    out_dtype = _out_dtype(hcube)
    nC = int(np.prod(hcube.shape[1:-2]))
    cubes = hcube.reshape((nT, nC, ny, nx))
    rot_hcube = np.empty(hcube.shape, dtype=out_dtype)
    for i1, i2 in chunk_ranges(nT, 10*nC*ny*nx*np.dtype(out_dtype).itemsize, _MAX_CHUNK_BYTES):
        t = to_tensor(cubes[i1:i2], device, dtype)
        x, y = _rotation_sampling_coords((ny, nx), angles[i1:i2], cent, new_cent, dtype, device)
        rot = _interp_propagating(t, x, y, 3, cval0, 0.5, 0., dilate=False)
        rot_hcube[i1:i2] = to_numpy(rot, out_dtype).reshape((i2-i1, *hcube.shape[1:]))
    unrotated = angles == 0.
    rot_hcube[unrotated] = hcube[unrotated] # Exposures that are not rotated are returned as they were rather than interpolated
    return rot_hcube


def pad_and_rotate_hypercube(hcube, angles, cent=None, device=None, cval0=np.nan):
    ny, nx = hcube.shape[-2:]
    if cent is None:
        cent = (np.array([nx, ny])-1.)/2.
    dymin_pad, dymax_pad, dxmin_pad, dxmax_pad = compute_derot_padding(nx, ny, angles, cent=cent)
    hcube_pad = np.pad(hcube.copy(), [*[[0,0] for _ in range(hcube.ndim-2)], [dymin_pad, dymax_pad], [dxmin_pad, dxmax_pad]], constant_values=np.nan)
    cent_pad = cent + np.array([dxmin_pad, dymin_pad])
    hcube_pad_rot = rotate_hypercube(hcube_pad, angles, cent=cent_pad, device=device, cval0=cval0)
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


def deprojected_radial_profile(im, pa, incl, cent, pxscale, rmin, rmax, dr=0.1, rw=0.1):
    """
    Computes the mean deprojected radial profile.
    
    Parameters
    ----------
    im: ndarray
        Image for which to measure the profile. NaNs in the array will result
        in NaNs in the output profile.

    pa: float
        Position angle for the deprojection in degrees.
        
    incl: float
        Inclination for the deprojection in degrees.
        
    cent: ndarray
        Two element array providing the center point for the profile as [x,y].
        
    pxscale: float or astropy.units.quantity.Quantity
        The pixel scale for the data. If a unitless float, value is assumed to be
        arcsec/pixel. If having astropy units, the units must be equivalent to 
        arcsec/pixel (e.g., mas/pixel, deg/pixel).
        
    rmin: float
        Minimum radius for measurements in arcsec.
        
    rmax: float
        Maximum radius for measurements in arcsec.
        
    dr: float
        Radial spacing for measurements in arcsec.
    
    rw: float
        Radial width of annuli in arcsec.

    Returns
    -------
    rads: ndarray
        Average radius of each measurement in arcsec.
        
    rprof: ndarray
        Deprojected radial profile having the same units as the input image.
    """
    irads = np.arange(rmin, rmax+dr, dr) # inner radii in arcsec
    orads = irads + rw # outer radii in arcsec
    rads = 0.5*(irads+orads) # average radii for each annulus
    cosi = np.cos(np.deg2rad(incl)) # cosine of the inclination
    
    rprof = np.zeros(irads.shape[0], dtype=np.float32)
    for i,(irad,orad) in enumerate(zip(irads,orads)):
        if irad == 0: irad += 1e-12 # irad = 0 otherwise raises an error
        aper = EllipticalAnnulus(cent, a_in=ang_size_to_px_size(irad, pxscale).value, a_out=ang_size_to_px_size(orad, pxscale).value,
                                 b_in=cosi*ang_size_to_px_size(irad, pxscale).value, b_out=cosi*ang_size_to_px_size(orad, pxscale).value,
                                 theta=(pa+90)*u.deg)
        rprof[i] = np.array(aperture_photometry(im, aper)['aperture_sum'])/aper.area
    return rads, rprof


def radial_calc_1d(im, cent, fn, fn_kwargs={}, dr=0.5, rmin=None, rmax=None, feature_mask=None, return_rmap=False, dtype=np.float32):
    """
    Computed a radial profile for 'im'
    """
    ny, nx = im.shape
    rmap = dist_to_pt(cent, nx, ny, dtype=np.float32)
    if rmin is None: rmin = np.min(rmap)
    if rmax is None: rmax = np.max(rmap)
    rarr = np.arange(rmin, rmax+dr, dr, dtype=np.float32)
    rprof_1d = np.zeros_like(rarr)+np.nan
    im_prepped = im.copy()
    if feature_mask is not None:
        im_prepped[feature_mask] = np.nan
    nans_msk = ~np.isfinite(im_prepped)
    for i, rv in enumerate(rarr):
        ropt = (rmap <= rv + dr) & (rmap >= rv - dr)
        if np.any(ropt):
            fopt = im_prepped[ropt & ~nans_msk]
            rprof_1d[i] = fn(fopt, **fn_kwargs)
    out = [rarr, rprof_1d]
    if return_rmap:
        out.append(rmap)
    return out


def radial_calc_2d(im, cent, fn, fn_kwargs={}, dr=0.5, rmin=None, rmax=None, feature_mask=None, dtype=np.float32):
    from scipy.interpolate import interp1d
    out_1d = radial_calc_1d(im, cent, fn, fn_kwargs, dr, rmin, rmax, feature_mask, return_rmap=True, dtype=dtype)
    rarr, rprof_1d, rmap = out_1d[0], out_1d[1], out_1d[-1]
    rprof_2d = interp1d(rarr, rprof_1d, fill_value=np.nan, bounds_error=False)(rmap)
    return rprof_2d


def disk_profile_xy(r, incl, pa, dphi=1.0, return_phi=False):
    """
    Returns cartesian coordinates (and optionally: azimuthal angles in degrees)
    for positions along a circle of radius r with incl (deg) and pa (deg),
    placed every dphi degrees. Output units are the same as those of r (except
    phi, which is in degrees).
    """
    incl_rad, pa_rad = np.deg2rad(incl), np.deg2rad(pa)
    phi = np.arange(0.0, 360.0, dphi, dtype=np.float32)
    phi_rad = np.deg2rad(phi)
    X0, Y0 = r*np.sin(phi_rad), -r*np.cos(phi_rad)*np.cos(incl_rad)
    X = X0*np.sin(np.pi - pa_rad) + Y0*np.cos(np.pi - pa_rad)
    Y = X0*np.cos(np.pi - pa_rad) - Y0*np.sin(np.pi - pa_rad)
    if return_phi:
        return X, Y, phi
    return X, Y


def setup_jupyter_display(width=95, fontsize=16):
    """
    Sets window width and markdown fontsize for Jupyter notebook. Width is % of window.
    """
    from IPython.display import display, HTML
    display(HTML("<style>.container { width:"+str(width)+"% !important; }</style>"))
    display(HTML("<style>.rendered_html { font-size: "+str(fontsize)+"px; }</style>"))
    return None


def bin_image(im, binfac=3, err=None, keep_shape=False):
    """
    Bin an image (and, optionally, its uncertainty map) by an integer factor.

    Note: this assumes that the uncertainties are uncorrelated and so resulting
    uncertainties should be assumed to be underestimated otherwise.

    Parameters
    ----------
    im : 2D numpy.ndarray
        Input image.
    err : 2D numpy.ndarray
        Input uncertainty map
    binfac : int
        Binning factor along Y and X (assumed square bins).
    keep_shape : bool
        If True, images are returned with the original shape but with all pixels in a given bin
        set to the same value.

    Returns
    -------
    im_bin : 2D numpy.ndarray
        Binned image (pixel values are means).
    err_bin : 2D numpy.ndarray
        Binned uncertainty map (standard error of the mean).
    """

    ny0, nx0 = im.shape

    if ny0 % binfac != 0:
        ypadfull = binfac - (ny0 % binfac)
    else:
        ypadfull = 0

    if nx0 % binfac != 0:
        xpadfull = binfac - (nx0 % binfac)
    else:
        xpadfull = 0

    ypad = [0, ypadfull]
    xpad = [0, xpadfull]

    im_pad = np.pad(im, [ypad, xpad], constant_values=np.nan)
    ny, nx = im_pad.shape

    # Reshape into (n_y/binfac, binfac, n_x/binfac, binfac)
    newy = ny // binfac
    newx = nx // binfac

    binningshape = (newy, binfac, newx, binfac)
    im_reshaped  = im_pad.reshape(binningshape)

    # Mean of the values
    im_bin = np.mean(im_reshaped, axis=(1,3), keepdims=keep_shape)
    if keep_shape:
        im_bin = np.broadcast_to(im_bin, binningshape).reshape((ny0+ypadfull), (nx0+xpadfull))[:ny0, :nx0]
        
    if err is None:
        return im_bin
    
    # Standard error propagation: σ_mean = sqrt(sum σ_i^2) / N
    # where N = binfac*binfac
    err_pad = np.pad(err, [ypad, xpad], constant_values=np.nan)
    unity = np.pad(np.ones_like(im), [ypad, xpad], constant_values=np.nan)

    err_reshaped = err_pad.reshape(binningshape)
    uni_reshaped = unity.reshape(binningshape)
    
    N = np.nansum(uni_reshaped, axis=(1,3), keepdims=keep_shape)
    
    err_bin = np.sqrt(np.nansum(err_reshaped**2, axis=(1,3), keepdims=keep_shape))/N

    if keep_shape:
        err_bin = np.broadcast_to(err_bin, binningshape).reshape((ny0+ypadfull), (nx0+xpadfull))[:ny0, :nx0]

    return im_bin, err_bin


def source(fn):
    """
    Prints the source code for the function fn.
    """
    import inspect
    print(inspect.getsource(fn))
    return None

