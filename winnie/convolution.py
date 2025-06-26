import numpy as np
import webbpsf_ext
from jwst.coron import imageregistration
from astropy.convolution.kernels import (Gaussian2DKernel, _round_up_to_odd_integer)
from .utils import (pad_or_crop_image, xy_polar_ang_displacement, c_to_c_osamp, rotate_image)
import astropy.units as u
from scipy import signal
from joblib import Parallel, delayed
from copy import deepcopy
from tqdm.auto import tqdm


def convolve_with_spatial_psfs(im_in, psfs, psf_inds, coron_tmap=None, use_gpu=False, ncores=-2):
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

        coron_tmap: ndarray, optional
            2D array of coronagraph transmission (same shape as im_in), by
            which im_in will be multiplied before convolution.
        
        use_gpu: bool, optional
            If True, use faster GPU-based CuPy routines for convolution.
            
        ncores: int, optional
            The number of CPU cores to use (when use_gpu=False) for convolution
            
    Returns:
        imcon: ndarray
            Convolved image of the same shape as im_in.
    """
    im = im_in.copy()
    if coron_tmap is not None:
        im *= coron_tmap
        
    convolution_fn = psf_convolve_gpu if use_gpu else psf_convolve_cpu
    
    yi,xi = np.indices(im.shape)
    nonzero = im != 0.
    
    psf_yhw, psf_xhw = np.ceil(np.array(psfs.shape[-2:])/2.).astype(int)
    xi_nz, yi_nz = xi[nonzero], yi[nonzero]
    x1, x2 = int(max(xi_nz.min()-psf_xhw, 0)), int(min(xi_nz.max()+psf_xhw, im.shape[-1]))
    y1, y2 = int(max(yi_nz.min()-psf_yhw, 0)), int(min(yi_nz.max()+psf_yhw, im.shape[-2]))
    
    im_crop = im[y1:y2+1, x1:x2+1]
    psf_inds_crop = psf_inds[y1:y2+1, x1:x2+1]
    
    if use_gpu or ncores==1:
        imcon_crop = np.zeros(im_crop.shape, dtype=im.dtype)
        for i in np.unique(psf_inds_crop):
            im_to_convolve = np.where(psf_inds_crop==i, im_crop, 0.)
            imcon_crop += convolution_fn(im_to_convolve, psfs[i])
    else:
        imcon_crop = np.sum(Parallel(n_jobs=ncores, prefer='threads')(delayed(convolution_fn)(np.where(psf_inds_crop==i, im_crop, 0.), psfs[i]) for i in np.unique(psf_inds_crop)), axis=0)
        
    imcon = np.zeros_like(im)
    imcon[y1:y2+1, x1:x2+1] = imcon_crop
    return imcon


def psf_convolve_gpu(im, psf_im):
    """
    GPU-based PSF convolution using CuPy's version of scipy.signal's fast
    fourier transform.
    """
    imcon = cp.asnumpy(cp_signal.fftconvolve(cp.array(im), cp.array(psf_im), mode='same'))
    return imcon


def psf_convolve_cpu(im, psf_im):
    """
    CPU-based PSF convolution using scipy.signal's fast fourier transform.
    """
    imcon = signal.fftconvolve(im, psf_im, mode='same')
    return imcon


def generate_lyot_psf_grid(inst, source_spectrum=None, nr=12, ntheta=4, log_rscale=True, rmin=0.05, rmax=3.5, normalize='exit_pupil', shift=None, osamp=2, fov_pixels=201, show_progress=True):
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

    if source_spectrum is None:
        source_spectrum = stpsf.specFromSpectralType('G2V')

    if isinstance(source_spectrum, dict): # Assume it's already a dictionary of weights
        source_weights = source_spectrum
    else:
        import stpsf
        nlambda = inst._get_default_nlambda(inst.filter)
        if isinstance(inst, stpsf.stpsf_core.NIRCam):
            if inst.pupil_mask.endswith('RND'): pupil_mask_ext = 'CIRCLYOT'
            elif inst.pupil_mask.endswith('WB'): pupil_mask_ext = 'WEDGELYOT'
            else: pupil_mask_ext = inst.pupil_mask
            bandpass_ext = webbpsf_ext.bandpasses.nircam_filter(inst.filter, pupil=pupil_mask_ext, mask=inst.image_mask, module=inst.module, sca=inst.detector)
        else:
            bandpass_ext = webbpsf_ext.bandpasses.miri_filter(inst.filter)
        source_weights = source_spectrum_to_weights(source_spectrum, bandpass_ext, nlambda)

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
    inst_grid = deepcopy(inst)
    iterator = tqdm(psf_offsets_in.T, leave=False) if show_progress else psf_offsets_in.T
    for psf_offset in iterator:
        inst_grid.options['coron_shift_x'] = -psf_offset[0]
        inst_grid.options['coron_shift_y'] = -psf_offset[1]
        psf = inst_grid.calc_psf(source=source_weights, fov_pixels=fov_pixels, oversample=osamp, normalize=normalize)[2].data
        psfs.append(psf)
    psfs = np.array(psfs)
    
    extra_shift = np.array([psfs.shape[1]%2-1, psfs.shape[2]%2-1])[::-1]/2.

    if shift is not None:
        psfs = np.array([imageregistration.fourier_imshift(psf, shift*osamp+extra_shift) for psf in psfs])
        
    if np.any(extra_shift != 0): # Crop any even axes to odd
        psfs = psfs[:, :int(extra_shift[1]*2), :int(extra_shift[0]*2)]

    # Science positions in detector pixels
    field_rot = 0 if inst._rotation is None else inst._rotation
    
    psf_offsets_polar = np.array([rvals_all, thvals_all-field_rot])
    psf_offsets = np.array(webbpsf_ext.coords.rtheta_to_xy(*psf_offsets_polar))
    
    return psfs, psf_offsets_polar, psf_offsets


def get_stpsf_model_center_offset(psf_off, osamp, x_sfac=0.25, y_sfac=0.50):
    """
    Returns the detector-sampled shift required to geometrically center psf_off,
    a PSF model generated with image_mask=None and with oversampling osamp.
    """
    x_stddev, y_stddev = x_sfac*osamp, y_sfac*osamp
    x_size = y_size = (15 if _round_up_to_odd_integer(8*max(x_stddev,y_stddev))<15 else None)
    psf_gauss = Gaussian2DKernel(x_stddev=x_stddev, y_stddev=y_stddev, x_size=x_size, y_size=y_size).array
    psf_gauss *= psf_off.max() / psf_gauss.max()
    psf_crop = pad_or_crop_image(psf_off, psf_gauss.shape, cval0=0)
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
            c_star=c_coron
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


def get_jwst_coron_transmission_map(inst, c_coron, return_oversample=True, osamp=None, nd_squares=True, shape=None, posang=0, c_star=None):
    """
    Generates a coronagraph transmission map relative to position c_coron
    (detector sampled [x,y] in pixels) using inst — a WebbPSF-ext instrument
    object. 

    If posang and c_star are provided, generates the same map but assuming the
    data have been derotated for angle 'posang' about position 'c_star'. 
    """

    if shape is None:
        ny, nx = inst.siaf_ap.YSciSize, inst.siaf_ap.XSciSize
    else:
        ny, nx = shape

    if osamp is None:
        osamp = inst.oversample
    elif osamp != inst.oversample:
        inst.oversample = osamp
        
    if isinstance(inst, webbpsf_ext.webbpsf_ext_core.MIRI_ext):
        im_mask_osamp = inst.gen_mask_image(npix=max(nx,ny)*osamp*2, pixelscale=inst.pixelscale/osamp)
    else:
        im_mask_osamp = inst.gen_mask_image(npix=max(ny,nx)*osamp*2, nd_squares=nd_squares, pixelscale=inst.pixelscale/osamp)

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


try:
    import cupy as cp # type: ignore
    from cupyx.scipy import signal as cp_signal # type: ignore
    gpu = cp.cuda.Device(0)
except ModuleNotFoundError:
    pass