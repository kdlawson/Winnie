import numpy as np
import torch
import webbpsf_ext
from jwst.coron import imageregistration
from astropy.convolution.kernels import (Gaussian2DKernel, _round_up_to_odd_integer)
from .utils import (pad_or_crop_image, xy_polar_ang_displacement, c_to_c_osamp, rotate_image)
from .backend import get_device, resolve_dtype, to_tensor, to_numpy, chunk_ranges
import astropy.units as u
from concurrent.futures import ThreadPoolExecutor
from scipy.fft import next_fast_len
from copy import deepcopy
from tqdm.auto import tqdm
from astropy.io import fits

# On the CPU, localized convolution (which is run in parallel over the PSF regions) is used when method='auto' if its
# estimated FFT work is less than this fraction of that for full-frame convolution.
_AUTO_CONVOLUTION_MARGIN = 0.9


def _get_psf_patch_bounds(psf_inds, n_psfs, psf_hws):
    """
    Compute padded bounding boxes for each PSF index present in ``psf_inds``.
    
    Parameters
    ----------
    psf_inds : ndarray
        2D array of PSF indices.
    n_psfs : int
        Number of PSFs.
    psf_hws : tuple
        (psf_height_half_width, psf_width_half_width).
    
    Returns
    -------
    used_inds : ndarray
        Indices of PSFs present in ``psf_inds``.
    bounds : ndarray
        Padded bounding boxes as (y1, y2, x1, x2) for each PSF index.
    """
    ny, nx = psf_inds.shape
    flat_inds = np.ravel(psf_inds)
    counts = np.bincount(flat_inds, minlength=n_psfs)
    used_inds = np.flatnonzero(counts)

    y_coords, x_coords = [i.ravel() for i in np.indices((ny,nx), dtype=np.int32)]

    min_y = np.full(n_psfs, ny, dtype=np.int32)
    min_x = np.full(n_psfs, nx, dtype=np.int32)
    max_y = np.full(n_psfs, -1, dtype=np.int32)
    max_x = np.full(n_psfs, -1, dtype=np.int32)

    np.minimum.at(min_y, flat_inds, y_coords)
    np.minimum.at(min_x, flat_inds, x_coords)
    np.maximum.at(max_y, flat_inds, y_coords)
    np.maximum.at(max_x, flat_inds, x_coords)

    psf_yhw, psf_xhw = psf_hws
    bounds = np.column_stack((
        np.maximum(min_y[used_inds] - psf_yhw, 0),
        np.minimum(max_y[used_inds] + psf_yhw, ny - 1),
        np.maximum(min_x[used_inds] - psf_xhw, 0),
        np.minimum(max_x[used_inds] + psf_xhw, nx - 1),
    ))
    return used_inds, bounds


def _estimate_fft_convolution_cost(im_shapes, psf_shape):
    """
    Estimate relative FFT convolution work for one or more image shapes.
    """
    im_shapes = np.atleast_2d(np.asarray(im_shapes, dtype=np.float64))
    conv_sizes = np.prod(im_shapes + np.asarray(psf_shape, dtype=np.float64) - 1, axis=1)
    return conv_sizes * np.log2(np.maximum(conv_sizes, 2.0))


def _choose_spatial_psf_convolution_strategy(im_shape, psf_shape, patch_bounds, return_costs=False):
    """
    Choose between full-frame and localized convolution based on FFT work.
    """
    if len(patch_bounds) == 0:
        return 'localized'

    full_cost = len(patch_bounds) * _estimate_fft_convolution_cost(im_shape, psf_shape)[0]
    patch_shapes = patch_bounds[:, [1, 3]] - patch_bounds[:, [0, 2]] + 1
    localized_cost = _estimate_fft_convolution_cost(patch_shapes, psf_shape).sum()
    if return_costs:
        return full_cost, localized_cost
    return 'localized' if localized_cost < (full_cost * _AUTO_CONVOLUTION_MARGIN) else 'full'


def _fft_conv_shape(im_shape, psf_shape):
    """
    Shape of the FFTs needed to convolve an image and a PSF without wrap-around.
    """
    return tuple(next_fast_len(int(n + p - 1), real=True) for n, p in zip(im_shape, psf_shape))


class SpatialPSFConvolver:
    """
    Convolves images with a stack of PSFs, using the PSF indicated by psf_inds for each pixel of the image (i.e., a
    shift-variant convolution):

        out = sum_i convolve(im * (psf_inds == i), psfs[i], mode='same')

    The convolutions are carried out in the Fourier domain on the chosen device. The Fourier transforms of the PSFs are
    cached, so repeated convolutions with the same PSFs (e.g., during deconvolution or when fitting models) only require
    transforming the images.

    Parameters
    ----------
    psfs: ndarray
        3D array of PSFs with shape (nPSF, ny_psf, nx_psf), or a single 2D PSF.

    device: str or torch.device or None
        Device on which to carry out the convolutions. If None, the default device is used (see winnie.backend).

    max_batch_bytes: int
        Approximate maximum memory to use for temporary arrays when transforming multiple PSFs and image regions at once.

    max_cache_bytes: int
        Maximum memory to use for caching the Fourier transforms of the PSFs. Transforms that would exceed this are
        recomputed for each convolution instead.

    Notes
    -----
    Calculations are carried out in the highest precision supported by the device that is warranted by the dtypes of the
    PSFs and the image (float32 on 'mps', which does not support float64). Numpy inputs yield numpy outputs with the dtype
    of the input image, and torch tensors yield tensors. On the CPU, localized convolutions are run in parallel over
    torch.get_num_threads() threads (see the ncores argument of SpaceRDI).
    """
    def __init__(self, psfs, device=None, max_batch_bytes=2**30, max_cache_bytes=2**31):
        psfs = np.asarray(psfs)
        self.psfs = psfs[np.newaxis] if psfs.ndim == 2 else psfs
        self.device = get_device(device)
        self.max_batch_bytes = max_batch_bytes
        self.max_cache_bytes = max_cache_bytes
        self._full_ffts = {}
        self._patch_ffts = {}
        self._cache_bytes = 0

    def __call__(self, im, psf_inds=None, coron_tmap=None, method='auto', used_inds=None, patch_bounds=None):
        """
        Convolves the 2D image im. See convolve_with_spatial_psfs for the arguments. If there is only a single PSF,
        psf_inds is not required.
        """
        method = method.lower()
        if method not in {'auto', 'localized', 'full'}:
            raise ValueError("method must be one of 'auto', 'localized', or 'full'")
        n_psfs = len(self.psfs)
        if psf_inds is None and n_psfs != 1:
            raise ValueError("psf_inds must be provided when convolving with multiple PSFs.")

        dtype = resolve_dtype(self.device, im.dtype, self.psfs.dtype)
        t = to_tensor(im, self.device, dtype)
        if coron_tmap is not None:
            t = t * to_tensor(coron_tmap, self.device, dtype)

        if psf_inds is None:
            inds = None
            out = self._convolve_full(t, inds, np.array([0]), dtype)
        else:
            inds = to_tensor(psf_inds, self.device)
            if method in {'auto', 'localized'} and ((used_inds is None) or (patch_bounds is None)):
                psf_yhw, psf_xhw = np.ceil(np.array(self.psfs.shape[-2:])/2.).astype(int)
                used_inds, patch_bounds = _get_psf_patch_bounds(to_numpy(inds), n_psfs, (psf_yhw, psf_xhw))
            if method == 'auto':
                # The overhead of preparing new FFT shapes is high on GPUs, so we only consider localized convolution on the CPU
                method = (_choose_spatial_psf_convolution_strategy(t.shape, self.psfs.shape[-2:], patch_bounds)
                          if self.device.type == 'cpu' else 'full')
            if method == 'full':
                if used_inds is None:
                    used_inds = np.flatnonzero(np.bincount(to_numpy(inds).ravel(), minlength=n_psfs))
                out = self._convolve_full(t, inds, used_inds, dtype)
            else:
                out = self._convolve_localized(t, inds, used_inds, patch_bounds, dtype)

        if isinstance(im, torch.Tensor):
            return out
        return to_numpy(out, im.dtype if im.dtype.kind == 'f' else np.float64)

    def _convolve_full(self, t, inds, used_inds, dtype):
        ny, nx = t.shape
        ph, pw = self.psfs.shape[-2:]
        used_inds = np.asarray(used_inds)
        if len(used_inds) == 0:
            return torch.zeros_like(t)
        fft_shape = _fft_conv_shape((ny, nx), (ph, pw))
        bytes_per_psf = 4 * fft_shape[0] * fft_shape[1] * t.element_size()
        acc = None
        for a, b in chunk_ranges(len(used_inds), bytes_per_psf, self.max_batch_bytes):
            ids = used_inds[a:b]
            if inds is None:
                parts = t[None]
            else:
                parts = torch.where(inds[None] == to_tensor(ids, self.device)[:, None, None], t[None], torch.zeros((), dtype=dtype, device=self.device))
            prod = (torch.fft.rfft2(parts, s=fft_shape) * self._psf_ffts(ids, fft_shape, dtype)).sum(0)
            acc = prod if acc is None else acc.add_(prod)
        out = torch.fft.irfft2(acc, s=fft_shape) # Summing in the Fourier domain means a single inverse transform is needed
        sy, sx = (ph-1)//2, (pw-1)//2
        return out[sy:sy+ny, sx:sx+nx]

    def _convolve_localized(self, t, inds, used_inds, patch_bounds, dtype):
        ph, pw = self.psfs.shape[-2:]
        sy, sx = (ph-1)//2, (pw-1)//2

        def convolve_patch(args):
            i, (y1, y2, x1, x2) = args
            h, w = y2-y1+1, x2-x1+1
            fft_shape = _fft_conv_shape((h, w), (ph, pw))
            patch = t[y1:y2+1, x1:x2+1].masked_fill(inds[y1:y2+1, x1:x2+1] != int(i), 0.)
            patch_conv = torch.fft.irfft2(torch.fft.rfft2(patch, s=fft_shape) * self._patch_psf_fft(int(i), fft_shape, dtype), s=fft_shape)
            return patch_conv[sy:sy+h, sx:sx+w]

        # The FFT of a single patch does not use many CPU threads, but torch releases the GIL, so the patches are
        # convolved in parallel. This is not done for GPUs, which are already highly parallel.
        args = list(zip(used_inds, patch_bounds))
        n_threads = torch.get_num_threads() if self.device.type == 'cpu' else 1
        if n_threads > 1 and len(args) > 1:
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                patches_conv = list(pool.map(convolve_patch, args))
        else:
            patches_conv = map(convolve_patch, args)
        out = torch.zeros_like(t)
        for patch_conv, (_, (y1, y2, x1, x2)) in zip(patches_conv, args):
            out[y1:y2+1, x1:x2+1] += patch_conv
        return out

    def _psf_ffts(self, ids, fft_shape, dtype):
        """
        Returns the Fourier transforms of the PSFs indicated by ids (sorted), zero-padded to fft_shape, using the cache
        for the whole PSF stack where possible.
        """
        key = (fft_shape, dtype)
        if key not in self._full_ffts:
            nbytes = len(self.psfs) * fft_shape[0] * (fft_shape[1]//2 + 1) * 2 * (torch.finfo(dtype).bits // 8)
            if self._cache_bytes + nbytes > self.max_cache_bytes:
                self._full_ffts[key] = None
            else:
                chunks = [torch.fft.rfft2(to_tensor(self.psfs[a:b], self.device, dtype), s=fft_shape)
                          for a, b in chunk_ranges(len(self.psfs), 4*fft_shape[0]*fft_shape[1]*(torch.finfo(dtype).bits // 8), self.max_batch_bytes)]
                self._full_ffts[key] = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
                self._cache_bytes += nbytes
        ffts = self._full_ffts[key]
        if ffts is None:
            return torch.fft.rfft2(to_tensor(self.psfs[ids], self.device, dtype), s=fft_shape)
        if ids[-1] - ids[0] + 1 == len(ids): # contiguous PSFs can be sliced without copying
            return ffts[ids[0]:ids[-1]+1]
        return ffts[to_tensor(ids, self.device)]

    def _patch_psf_fft(self, i, fft_shape, dtype):
        key = (i, fft_shape, dtype)
        fft = self._patch_ffts.get(key)
        if fft is None:
            fft = torch.fft.rfft2(to_tensor(self.psfs[i], self.device, dtype), s=fft_shape)
            nbytes = fft.numel() * fft.element_size()
            if self._cache_bytes + nbytes <= self.max_cache_bytes:
                self._patch_ffts[key] = fft
                self._cache_bytes += nbytes
        return fft


def convolve_with_spatial_psfs(im_in, psfs, psf_inds, patch_bounds=None, used_inds=None, coron_tmap=None, method='auto', device=None):
    """
    Creates a PSF-convolved image where each pixel of the input image has been
    convolved with the nearest spatially-sampled PSF.

    ___________
    Parameters:

        im_in: ndarray
            2D image array to be convolved.

        psfs: ndarray
            3D image array of spatially-sampled PSFs with which to convolve
            im_in. Generally, each

        psf_inds: ndarray
            2D array (same shape as im_in; dtype=int); each entry gives the
            index of the slice in psfs with which that pixel in im_in should be
            convolved.

        patch_bounds: ndarray, optional
            Pre-computed padded bounding boxes for each PSF index in psf_inds,
            as returned by _get_psf_patch_bounds. If not provided, will be
            computed on the fly if method is 'auto' or 'localized'.

        used_inds: ndarray, optional
            Pre-computed array of PSF indices present in psf_inds. If not
            provided, will be computed on the fly.

        coron_tmap: ndarray, optional
            2D array of coronagraph transmission (same shape as im_in), by
            which im_in will be multiplied before convolution.

        method: {'auto', 'localized', 'full'}, optional
            Convolution strategy to use. 'localized' uses padded bounding
            boxes around each PSF region, 'full' uses the full science
            frame for every PSF index (batched together), and 'auto' selects
            between them using an FFT work estimate (when running on the CPU;
            'full' is always used on GPUs).

        device: str or torch.device, optional
            Device on which to carry out the convolution (e.g., 'cpu', 'mps',
            'cuda'). If None, the default device is used (see winnie.backend).

        For repeated convolutions with the same PSFs, it is more efficient to
        create a SpatialPSFConvolver and call it directly, since this caches
        the Fourier transforms of the PSFs.

    Returns:
        imcon: ndarray
            Convolved image of the same shape as im_in.
    """
    return SpatialPSFConvolver(psfs, device=device)(im_in, psf_inds, coron_tmap=coron_tmap, method=method,
                                                    used_inds=used_inds, patch_bounds=patch_bounds)


def psf_convolve(im, psf_im, device=None):
    """
    Convolves the 2D image im with the single PSF (or kernel) psf_im using FFTs on device. The output has the same shape
    as im (equivalent to scipy.signal.fftconvolve with mode='same').
    """
    return SpatialPSFConvolver(np.asarray(psf_im), device=device)(im)


def generate_lyot_psf_grid(inst, 
                           source_spectrum=None, 
                           nr=12, 
                           ntheta=4, 
                           log_rscale=True, 
                           rmin=0.05, 
                           rmax=3.5, 
                           normalize='exit_pupil', 
                           shift=None, 
                           osamp=2, 
                           fov_pixels=201,
                           nlambda=None,
                           show_progress=True):
    """
    Creates a grid of synthetic PSFs using an STPSF NIRCam or MIRI object. The
    spatial sampling used here is not appropriate for non-Lyot data. Portions
    of this code are adapted from examples in the WebbPSF-ext documentation.
    
    ___________
    Parameters:

    inst: stpsf.stpsf_core.NIRCam or stpsf.stpsf_core.MIRI
        NIRCam or MIRI instrument object (set up appropriately for your data)
        to use for generating PSFs.
    
    source_spectrum: synphot spectrum, optional
        A synphot spectrum object to use when generating the PSFs.
        
    nr: int, optional
        The number of radial PSF samples to use in the grid. Actual grid will
        have nr+1 radial samples, since a grid point is added at r,theta =
        (0,0).
        
    ntheta: int, optional
        The number of azimuthal PSF samples to use in the grid.
        
    log_rscale: bool, optional
        If True (default), radial samples are logarithmically spaced (log10)
        between rmin and rmax arcseconds
        
    rmin: float, optional
        The minimum radial separation in arcseconds from the coronagraph center
        to use for the PSF grid.

    rmax: float, optional
        The maximum radial separation in arcseconds from the coronagraph center
        to use for the PSF grid.

    normalize: str, optional
        The normalization to use for the PSFs. Options are 'exit_pupil'
        (default), 'first', and 'last' (see STPSF documentation for more info).

    shift: ndarray, optional
        The detector sampled shift (in pixels) needed to place the PSF as the
        geometric center of the array. If None, no shift will be applied and
        the resulting PSFs may effectively shift your models during
        convolution.

    osamp: int, optional
        The oversampling factor for which to generate the synthetic PSFs

    fov_pixels: int, optional
        The number of pixels for each axis of the detector-sampled PSF models
        (returned images will be fov_pixels*osamp)

    show_progress: bool, optional
        If True, display a TQDM progress bar to track progress during PSF
        generation.
        
    Returns:
        psfs: ndarray
            The stack of synthetic PSF images
            
        psf_offsets_polar: ndarray
            An array of shape (2,Nsamples) providing the polar (r,theta) offset
            from the coronagraph center for each PSF sample in "psfs" in units
            of [arcsec, deg].

        psf_offsets: ndarray
            An array of shape (2,Nsamples) providing the (x,y) offset from the
            coronagraph center for each PSF sample in "psfs" in units of
            arcsec.

    Note: Currently, when computing weights from a source spectrum, STPSF's
    throughput for NIRCam a) assumes the mean of modules A and B, and b) does
    not account for the coronagraphic substrate transmission. For now, we're
    using a filter profile from WebbPSF-ext and computing our own weights
    instead. 
    """
    try:
        import stpsf
    except ModuleNotFoundError:
        import webbpsf as stpsf

    if source_spectrum is None:
        source_spectrum = stpsf.specFromSpectralType('G2V')

    if isinstance(source_spectrum, dict): # Assume it's already a dictionary of weights
        source_weights = source_spectrum
    else:
        if nlambda is None:
            nlambda = inst._get_default_nlambda(inst.filter)
        if isinstance(inst, stpsf.stpsf_core.NIRCam):
            if inst.pupil_mask.endswith('RND'): pupil_mask_ext = 'CIRCLYOT'
            elif inst.pupil_mask.endswith('WB'): pupil_mask_ext = 'WEDGELYOT'
            else: pupil_mask_ext = inst.pupil_mask
            bandpass_ext = webbpsf_ext.bandpasses.nircam_filter(inst.filter, pupil=pupil_mask_ext, mask=inst.image_mask, module=inst.module, sca=inst.detector)
        else:
            bandpass_ext = webbpsf_ext.bandpasses.miri_filter(inst.filter)
        source_weights = source_spectrum_to_weights(source_spectrum, bandpass_ext, nlambda)

    nlambda = len(source_weights['wavelengths'])
    
    # Set up the grid:
    if log_rscale:
        rvals = 10**(np.linspace(np.log10(rmin), np.log10(rmax), nr))
    else:
        rvals = np.linspace(rmin, rmax, nr)
        
    thvals = np.linspace(0, 360, ntheta, endpoint=False)
    rvals_all = [0]
    thvals_all = [0]
    for r in rvals:
        for th in thvals:
            rvals_all.append(r)
            thvals_all.append(th)
    rvals_all = np.array(rvals_all)
    thvals_all = np.array(thvals_all)
    xgrid_off, ygrid_off = webbpsf_ext.coords.rtheta_to_xy(rvals_all, thvals_all) # Mask Offset grid positions in arcsec

    psf_offsets_in = np.array([xgrid_off, ygrid_off])

    psfs = []
    psfs_nodist = []
    
    siaf_ap = inst.siaf[inst.aperturename]
    inst_grid = deepcopy(inst)

    iterator = tqdm(psf_offsets_in.T, leave=False) if show_progress else psf_offsets_in.T
    for psf_offset in iterator:
        inst_grid.options['coron_shift_x'] = -psf_offset[0]
        inst_grid.options['coron_shift_y'] = -psf_offset[1]

        inst_grid.detector_position = siaf_ap.idl_to_sci(*psf_offset)
        psf_hdul = inst_grid.calc_psf(source=source_weights, fov_pixels=fov_pixels, oversample=osamp, normalize=normalize, nlambda=nlambda)
        psfs.append(psf_hdul[2].data)

        psf_hdul = apply_partial_dist_to_stpsf_hdul(psf_hdul, inst_grid)
        psfs_nodist.append(psf_hdul[0].data)
   
    psfs = np.array(psfs)
    psfs_nodist = np.array(psfs_nodist)

    extra_shift = np.array([psfs.shape[1]%2-1, psfs.shape[2]%2-1])[::-1]/2.

    if shift is not None:
        psfs = np.array([imageregistration.fourier_imshift(psf, shift*osamp+extra_shift) for psf in psfs])
        psfs_nodist = np.array([imageregistration.fourier_imshift(psf, shift*osamp+extra_shift) for psf in psfs_nodist])

    if np.any(extra_shift != 0): # Crop any even axes to odd
        psfs = psfs[:, :int(extra_shift[1]*2), :int(extra_shift[0]*2)]
        psfs_nodist = psfs_nodist[:, :int(extra_shift[1]*2), :int(extra_shift[0]*2)]

    # Science positions in detector pixels
    field_rot = 0 if inst._rotation is None else inst._rotation
    
    psf_offsets_polar = np.array([rvals_all, thvals_all-field_rot])
    psf_offsets = np.array(webbpsf_ext.coords.rtheta_to_xy(*psf_offsets_polar))
    
    return psfs, psfs_nodist, psf_offsets_polar, psf_offsets


def get_stpsf_model_center_offset(psf_off, osamp, x_sfac=0.25, y_sfac=0.50, device=None):
    """
    Returns the detector-sampled shift required to geometrically center psf_off,
    a PSF model generated with image_mask=None and with oversampling osamp.
    The PyTorch device used for the interpolation is given by device (see winnie.backend).
    """
    x_stddev, y_stddev = x_sfac*osamp, y_sfac*osamp
    x_size = y_size = (15 if _round_up_to_odd_integer(8*max(x_stddev,y_stddev))<15 else None)
    psf_gauss = Gaussian2DKernel(x_stddev=x_stddev, y_stddev=y_stddev, x_size=x_size, y_size=y_size).array
    psf_gauss *= psf_off.max() / psf_gauss.max()
    psf_crop = pad_or_crop_image(psf_off, psf_gauss.shape, cval0=0, device=device)
    psf_reg_result = imageregistration.align_array(psf_gauss, psf_crop)
    shift = -psf_reg_result[1][:-1]/osamp
    return shift


def get_jwst_psf_grid_inds(c_coron, psf_offsets_polar, osamp=1, inst=None, shape=None, pxscale=None, posang=0, c_star=None):
    """
    Given a coronagraph center coordinate (c_coron), a list of PSF sample
    points relative to the coronagraph in polar coordinates ([r in arcsec,
    theta in deg]), and the desired oversampling factor (osamp), returns an
    image where each pixel value indicates the index of the nearest (in polar
    coordinates) spatial PSF sample from the PSF grid.

    Either inst or shape and pxscale must be provided. 'shape' and 'pxscale'
    should be the detector sampled dimensions (i.e., before oversampling) If
    inst is provided, assumes the nominal subarray size and scale for the
    output map.

    If posang and c_star are provided, generates the same map but assuming the
    data have been derotated for angle 'posang' about position 'c_star'. 
    """
    if shape is None:
        siaf_ap = inst.siaf[inst.aperturename]
        nx = siaf_ap.XSciSize
        ny = siaf_ap.YSciSize
    else:
        ny,nx = shape

    if pxscale is None:
        pxscale = inst.pixelscale*u.arcsec/u.pixel

    rvals_all, thvals_all = psf_offsets_polar.copy()

    rvals = np.unique(rvals_all)
    
    yg, xg = c_to_c_osamp(np.indices((ny*osamp, nx*osamp), dtype=np.float64), 1/osamp)

    psf_inds = np.zeros((ny*osamp, nx*osamp), dtype=np.int32)

    xmap_osamp, ymap_osamp = xg-c_coron[0], yg-c_coron[1]
    
    if posang != 0:
        if c_star is None:
            c_star = c_coron
        xmap_osamp, ymap_osamp = xy_polar_ang_displacement(xmap_osamp+c_coron[0]-c_star[0], ymap_osamp+c_coron[1]-c_star[1], -posang)
        xmap_osamp, ymap_osamp = xmap_osamp-c_coron[0]+c_star[0], ymap_osamp-c_coron[1]+c_star[1]
    
    rmap_osamp, tmap_osamp = webbpsf_ext.coords.xy_to_rtheta(xmap_osamp, ymap_osamp)
    tmap_osamp = np.mod(tmap_osamp, 360)
    rvals_px = rvals/pxscale.value
    
    nearest_rvals = rvals[np.argmin(np.array([np.abs(rmap_osamp-rval) for rval in rvals_px]), axis=0)]
    for rval in rvals:
        thvals = np.unique(thvals_all[rvals_all == rval])
        thvals_wrap0 = np.array([*thvals, *thvals])
        thvals_wrap = np.array([*thvals, *(thvals+360.)])
        nearest_thvals = thvals_wrap0[np.argmin(np.array([np.abs(tmap_osamp-thval) for thval in thvals_wrap]), axis=0)]
        for thval in thvals:
            i = np.where((thvals_all == thval)&(rvals_all == rval))[0][0]
            psf_inds[(nearest_rvals == rval)&(nearest_thvals == thval)] = i
    return psf_inds


def get_jwst_coron_transmission_map(inst_ext, c_coron, return_oversample=True, osamp=None, nd_squares=True, shape=None, posang=0, c_star=None):
    """
    Generates a coronagraph transmission map relative to position c_coron
    (detector sampled [x,y] in pixels) using inst — a WebbPSF-ext instrument
    object. 

    If posang and c_star are provided, generates the same map but assuming the
    data have been derotated for angle 'posang' about position 'c_star'. 
    """

    if shape is None:
        ny, nx = inst_ext.siaf_ap.YSciSize, inst_ext.siaf_ap.XSciSize
    else:
        ny, nx = shape

    if osamp is None:
        osamp = inst_ext.oversample
    elif osamp != inst_ext.oversample:
        inst_ext.oversample = osamp
        
    if isinstance(inst_ext, webbpsf_ext.webbpsf_ext_core.MIRI_ext):
        im_mask_osamp = inst_ext.gen_mask_image(npix=max(nx,ny)*osamp*2, pixelscale=inst_ext.pixelscale/osamp)
    else:
        im_mask_osamp = inst_ext.gen_mask_image(npix=max(ny,nx)*osamp*2, nd_squares=nd_squares, pixelscale=inst_ext.pixelscale/osamp)

    if posang != 0:
        if c_star is None:  
            im_mask_osamp = rotate_image(im_mask_osamp, -posang, cent=None, cval0=1)
        else:
            r_c_coron = c_coron-c_star
            im_mask_osamp = rotate_image(im_mask_osamp, -posang, cent=(np.array(im_mask_osamp.shape[::-1])-1)/2. - r_c_coron, cval0=1)
    
    im_mask_osamp = pad_or_crop_image(im_mask_osamp, new_size=[ny*osamp, nx*osamp], new_cent=c_to_c_osamp(c_coron, osamp), cval0=1, order=1)

    if return_oversample:
        return im_mask_osamp
    im_mask = webbpsf_ext.image_manip.frebin(im_mask_osamp, scale=1/osamp, total=False)
    return im_mask


def source_spectrum_to_weights(source_spectrum, bandpass_ext, nlambda):
    """
    Based on the input spectrum, computes weights to be used in combining
    monochromatic PSF slices.
    
    Note: This code is shamelessly copied nearly vertbatim from 
    the _calc_psf_with_shifts method in WebbPSF-ext.

    source_spectrum: synphot.spectrum.SourceSpectrum or webbpsf_ext.synphot_ext.Spectrum
        Input spectrum for the source.
        
    bandpass_ext: webbpsf_ext.synphot_ext.Bandpass
        Bandpass to use for generating weights.
        
    nlambda: int
        Number of linearly spaced wavelength samples within the bandpass at
        which to compute weights.
    """
    bpext_waves = bandpass_ext.waveset.to_value('um')[bandpass_ext.throughput > (0.01*bandpass_ext.throughput.max())]
    wave_um = np.linspace(bpext_waves.min(), bpext_waves.max(), nlambda)
    obs = webbpsf_ext.synphot_ext.Observation(source_spectrum, bandpass_ext, binset=wave_um*u.um)
    binflux = obs.sample_binned(flux_unit='counts').value
    weights = binflux / binflux.sum()
    return {'wavelengths': wave_um*1e-6, 'weights': weights}


def apply_partial_dist_to_stpsf_hdul(hdul, inst):
    """
    Applies all steps typically applied to an STPSF "OVERDIST" HDU to the
    'OVERSAMP' HDU except for astrometric distortion. The 'OVERSAMP' extension
    of the returned HDUList is then what we typically use for forward modeling
    in Winnie (where astrometric distortion is applied separately to the final
    image).
    """
    try:
        import stpsf
    except ModuleNotFoundError:
        import webbpsf as stpsf

    charge_diffusion_options = dict(charge_diffusion_sigma=inst.options.get('charge_diffusion_sigma', None))
    add_ipc = inst.options.get('add_ipc', True)
    # For NIRCam, the standard procedure is rotation >> distortion >> charge
    # diffusion >> IPC (leaving in NIRISS/FGS functionality in case someone needs it)
    # For MIRI, it's distortion >> scattering >> charge diffusion >> IPC
    if inst.name in ['NIRCam', 'NIRISS', 'FGS']:
        hdul_pdist = stpsf.distortion.apply_rotation(fits.HDUList([hdul[0], hdul[0]]), crop=True)
    elif inst.name == 'MIRI':
        hdul_pdist = stpsf.detectors.apply_miri_scattering(fits.HDUList([hdul[0], hdul[0]]))
    else:
        raise ValueError(f"Unsupported instrument {inst.name}")
    
    # Charge diff and IPC are applied after astrometric distortion in STPSF,
    # but the difference is negligible unless astrometric distortion is huge
    hdul_pdist = stpsf.detectors.apply_detector_charge_diffusion(hdul_pdist, charge_diffusion_options)
    hdul_pdist[1].name = 'OVERDIST'
    if add_ipc:
        hdul_pdist = stpsf.detectors.apply_detector_ipc(hdul_pdist, extname='OVERDIST')
    hdul[0] = hdul_pdist[1]
    hdul[0].name = 'OVERSAMP'
    return hdul

