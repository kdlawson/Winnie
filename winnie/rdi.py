import numpy as np
import astropy.units as u
import scipy.linalg as linalg
from joblib import Parallel, delayed
from copy import copy, deepcopy
from tqdm.auto import tqdm

from .utils import (dist_to_pt, rotate_hypercube, pad_and_rotate_hypercube, free_gpu, ang_size_to_px_size, proj_sep_to_ang_size)

def rdi_residuals(hcube, hcube_ref, optzones, subzones, hcube_css=None, ref_mask=None, show_progress=False, posangs=None, cent=None, 
                  objective=False, zero_nans=False, use_gpu=False, ncores=-2, return_coeffs=False, coeffs_in=None, return_psf_model=False,
                  pad_before_derot=False, opt_smoothing_fn=None, opt_smoothing_kwargs={}, err_hcube=None, err_hcube_ref=None):
    """
    Performs RDI PSF subtraction using LOCI (Lafreniere et al. 2007). Using available keyword arguments, this function can be used to 
    perform:

    a) Model Constrained RDI (MCRDI; Lawson et al. 2022) — by passing in a convolved model of the circumstellar scene (CSS) via hcube_css 
    b) Polarimetry Constrained RDI (PCRDI; Lawson et al. 2022) — by passing a polarimetry-based total intensity estimate of the CSS via
       hcube_css
    c) RDI with high-pass filtering for coefficients — via opt_smoothing_fn and opt_smoothing_kwargs
    d) Greedy Disk Subtraction (GreeDS; Pairet et al. 2021) — by passing a previous iteration in via hcube_css
    e) Classical RDI — by passing in a fixed reference star scaling factor via coeffs_in.
    f) Forward modeling — by passing a convolved CSS model in as hcube and leaving the remaining arguments unchanged
    ___________
    Parameters:
    
        hcube: ndarray
            4D image array to be PSF-subtracted; shape of (nT, nL, ny, nx) where nT is the number of exposures/integrations, 
            nL is the number of wavelengths, and ny & nx are the number of pixels in each spatial dimension.
            
        hcube_ref: ndarray
            4D image array with shape (nT_ref, nL, ny, nx) where nT_ref is the number of reference
            exposures/integrations, and the remaining axes match those of hcube. 
            
        optzones: ndarray
            3D boolean array; for each slice, the target and reference images will be compared over any pixels with
            a value of True. The resulting coefficients will be used to perform PSF-subtraction over the region
            indicated by the corresponding entry in subzones.
            
        subzones: ndarray
            3D boolean array; for each slice, PSF subtraction will be performed only on pixels with
            a value of True. 
            
    _________
    Optional:
    
        hcube_css: ndarray
            4D array; same shape as hcube. hcube_css should provide an estimate of the circumstellar signal 
            in hcube, rotated to the appropriate posangs and convolved with the appropriate PSF.
            
        ref_mask: ndarray
            2D boolean array of shape (len(optzones), nT_ref) that indicates which reference images should be considered
            for which optimization regions. E.g., if ref_mask[i,j] is False, then for the ith optimization zone (optzones[i]),
            the jth reference image (hcube_ref[j]) will NOT be used for construction of the PSF model. This can be useful if
            some reference exposures have anomalous features that make them problematic for some regions while still being
            suitable for others; e.g., an image with a bright background source near the edge of the FOV may still be useful
            for nulling the PSF near the inner working angle.
            
        show_progress: bool
            If True, a status bar will be displayed for the major components of PSF subtraction.
            
        posangs: ndarray
            1D array giving the position angle (in degrees) of each exposure in hcube. If provided, the output 
            residual hypercube will be derotated accordingly.
            
        cent: ndarray
            The cartesian pixel coordinate (x,y) corresponding to the central star's position in hcube for the purpose 
            of derotation.
            
        objective: bool
            If True, the output array will be (hcube - hcube_css) - hcube_psfmodel, where hcube_psfmodel is 
            constructed by comparing hcube_ref to (hcube - hcube_css). If False, the output array will simply
            be hcube - hcube_psfmodel. If hcube_css is None, then setting this will have no effect. 
            
        zero_nans: bool
            If True, any nans in the optimization zones will be replaced with zeros for the procedure.
            
        use_gpu: bool
            If True, use faster GPU-based CuPy routines throughout.
            
        ncores: int
            The number of processor cores to use. Default value of -2 uses all but one available core.
            
        return_psf_model: bool
            If True, the PSF-model hcube matching hcube in shape is returned instead of the residuals hcube. Will not be
            derotated (even if posangs is specified).

        pad_before_derot: bool
            If True, prior to derotation, the residuals are padded to sufficient size to avoid loss of pixels. Note: 
            output dimensions will not match that of the input when this option is used. 

        return_coeffs: bool
            If True, returns only the array of PSF model coefficients.
        
        coeffs_in: ndarray
            If provided, these coefficients will be used to construct the PSF model instead of computing coefficients.
            
        opt_smoothing_fn: callable or None
            If not None, this argument indicates the function with which to smooth the sequences. This should
            be a function that takes a hypercube along with some keyword arguments and returns a smoothed hypercube, 
            i.e.: hcube_filt = opt_smoothing_fn(hcube, **opt_smoothing_kwargs).   
            Defaults to median_filter_sequence when opt_smoothing is set.
        
        opt_smoothing_kwargs: dict
            If opt_smoothing_fn is not None, arguments to pass to opt_smoothing_fn when it is called.

        err_hcube: ndarray or None
            If provided, this array should contain the uncertainties in the target hypercube. If None, the output
            err_hcube_res array will be None.

        err_hcube_ref: ndarray or None
            If provided, this array should contain the uncertainties in the reference hypercube. If None, but err_hcube
            is not None, the output err_hcube_res array will assume zero noise contribution from the PSF model (i.e., 
            for synthetic reference images).

    ________
    Returns:
        Default:
            hcube_res: ndarray
                4D array of PSF-subtracted residuals (derotated if 'posangs' was specified).
            err_hcube_res: ndarray or None
                4D array of uncertainties in the PSF-subtracted residuals (derotated if 'posangs' was specified). 
                If err_hcube is None, this will be None.
            cent_pad: tuple
                the center location with consideration for any added padding.
    """

    if hcube_css is None:
        hcube_sub = hcube
    else:
        hcube_sub = hcube - hcube_css
        
    if opt_smoothing_fn is not None:
        hcube_opt, hcube_ref_opt = opt_smoothing_fn(hcube_sub, **opt_smoothing_kwargs), opt_smoothing_fn(hcube_ref, **opt_smoothing_kwargs)
    else:
        hcube_opt, hcube_ref_opt = hcube_sub, hcube_ref

    if zero_nans: 
        hcube_opt = np.nan_to_num(hcube_opt)
        hcube_ref_opt = np.nan_to_num(hcube_ref_opt)
    
    if coeffs_in is None:
        coeffs = compute_rdi_coefficients(hcube_opt, hcube_ref_opt, optzones, show_progress=show_progress, ref_mask=ref_mask)
    else:
        coeffs = coeffs_in

    if return_coeffs:
        return coeffs

    if objective:
        psf_model = reconstruct_psf_model(hcube_ref_opt, coeffs, subzones, show_progress=show_progress, use_gpu=use_gpu, ncores=ncores)
    else:
        psf_model = reconstruct_psf_model(hcube_ref, coeffs, subzones, show_progress=show_progress, use_gpu=use_gpu, ncores=ncores)

    if return_psf_model:
        return psf_model
    
    if objective:
        hcube_res = hcube_opt - psf_model
    else:
        hcube_res = hcube - psf_model

    cent_out = copy(cent)
    if posangs is not None:
        if pad_before_derot:
            hcube_res, cent_out = pad_and_rotate_hypercube(hcube_res, -posangs, cent=cent, ncores=ncores, use_gpu=use_gpu, cval0=np.nan)
        else:
            hcube_res = rotate_hypercube(hcube_res, -posangs, cent=cent, ncores=ncores, use_gpu=use_gpu, cval0=np.nan)

    if err_hcube is None:
        err_hcube_res = None
    else: # TODO: outsource to reconstruct_psf_model to consider use_gpu / ncores to speed this up.
        err_hcube_psfmodel = np.zeros_like(err_hcube)
        if err_hcube_ref is not None:
            for i in range(hcube.shape[0]):
                for j in range(hcube.shape[1]):
                    lin_term_errs = (coeffs[j,0,i][:,np.newaxis,np.newaxis])*err_hcube_ref[:,j]
                    err_hcube_psfmodel[i,j] = np.sqrt(np.sum(lin_term_errs**2, axis=0))
        err_hcube_res = np.hypot(err_hcube, err_hcube_psfmodel)
        if posangs is not None:
            if pad_before_derot:
                err_hcube_res, _ = pad_and_rotate_hypercube(err_hcube_res, -posangs, cent=cent, ncores=ncores, use_gpu=use_gpu, cval0=np.nan)
            else:
                err_hcube_res = rotate_hypercube(err_hcube_res, -posangs, cent=cent, ncores=ncores, use_gpu=use_gpu, cval0=np.nan)
    if use_gpu:
        free_gpu()
    return hcube_res, err_hcube_res, cent_out


def compute_rdi_coefficients(hcube, hcube_ref, optzones, show_progress=False, ref_mask=None):
    nT, nL, _, _ = hcube.shape  # N_theta by N_lambda by N_y by N_x
    nT_ref = hcube_ref.shape[0]
    nR = len(optzones) # N_regions
    coeff_hcube = np.zeros((nL, nR, nT, nT_ref), dtype=hcube.dtype)  # Array for storing coefficients
    R_iterator = tqdm(range(nR), desc='Coefficient calculation (regions)', leave=False) if show_progress else range(nR)
    L_iterator = tqdm(range(nL), desc='Coefficient calculation (wavelengths)', leave=False) if (show_progress and nL>1) else range(nL)
    T_iterator = tqdm(range(nT), desc='Coefficient calculation (exposures)', leave=False) if show_progress else range(nT)
    for Ri in R_iterator:  # Outermost loop over subsections; this iteration order ends up being more time efficient here:
        opt_i = optzones[Ri]  # opt_i is the ny by nx boolean array indicating which pixels are in the region.
        tararrs = hcube[:, :, opt_i].copy() # This turns our masked nT*nL*ny*nx array into an nT*nL*nP array, where nP is the number of pixels in the optimization region.
        refarrs = hcube_ref[:, :, opt_i].copy()  # This turns our masked nT_ref*nL*ny*nx array into an nT_ref*nL*nP array, where nP is the number of pixels in the optimization region.
        if ref_mask is not None:
            refarrs = refarrs[ref_mask[Ri]]
        optmats = refarrs.transpose((1, 0, 2)) @ tararrs.transpose((1, 2, 0))# carries out matrix inversion of all wavelength channels at once, giving a matrix of shape (nL, nT_ref, nT)
        refmats = refarrs.transpose((1, 0, 2)) @ refarrs.transpose((1, 2, 0))# carries out matrix inversion of all wavelength channels at once, giving a matrix of shape (nL, nT_ref, nT_ref)
        for Li in L_iterator:  # Second loop over wavelengths:
            optmat = optmats[Li]  # The (nT_ref, nT) matrix for this wavelength
            refmat = refmats[Li]  # The (nT_ref, nT_ref) matrix for this wavelength
            lu, piv = linalg.lu_factor(refmat)  # Since we aren't excluding frames as in ADI/SDI, we just need to run this once per wavelength.
            for Ti in T_iterator:  # Final loop over par ang / time axis (individual science exposures):
                tararr = optmat[:,Ti]  # 1d vector of length equal to nT_ref
                if ref_mask is not None:
                    coeff_hcube[Li, Ri, Ti, ref_mask[Ri]] = linalg.lu_solve((lu, piv), tararr) # Gets coefficients and places them into the appropriate positions in the coefficient array
                else:
                    coeff_hcube[Li, Ri, Ti] = linalg.lu_solve((lu, piv), tararr) # Gets coefficients and places them into the appropriate positions in the coefficient array
    return coeff_hcube


def reconstruct_psf_model(hcube_ref, coeffs, subzones, show_progress=False, use_gpu=False, ncores=-2):
    if use_gpu:
        hcube_psfmodel = reconstruct_psf_model_gpu(hcube_ref, coeffs, subzones, show_progress=show_progress)
    else:
        hcube_psfmodel = reconstruct_psf_model_cpu(hcube_ref, coeffs, subzones, show_progress=show_progress, ncores=ncores)
    return hcube_psfmodel


def reconstruct_psf_model_cpu(hcube_ref, coeffs, subzones, show_progress=False, ncores=-2):
    _, _, ny, nx = hcube_ref.shape # Number of: images, wavelengths, y-pixels, and x-pixels
    nL, nR, nT, _ = coeffs.shape
    hcube_psfmodel = np.zeros((nT, nL, ny, nx), dtype=hcube_ref.dtype) + np.nan # Array in which to place reconstructed model
    for Ri in tqdm(range(nR), desc='PSF model reconstruction (regions)', leave=False) if show_progress else range(nR): # Iterate over regions
        sub_i = subzones[Ri] # An ny by nx boolean mask indicating which pixels are being considered
        imvals = hcube_ref[:, :, sub_i].T.copy() # Fetching the pixels in the subzone, dimensions of nT_ref x nL x npx
        T = tqdm(range(nT), desc='PSF model reconstruction (exposures)', leave=False) if show_progress else range(nT)
        nT_results = Parallel(n_jobs=ncores, prefer='threads')(delayed(np.sum)(coeffs[np.newaxis, :, Ri, Ti].copy()*imvals, -1) for Ti in T)
        for Ti in range(nT): # Iterate over target images, building the PSF model (in the subzone) for each.
            hcube_psfmodel[Ti, :, sub_i] = nT_results[Ti]
    return hcube_psfmodel


def reconstruct_psf_model_gpu(hcube_ref, coeffs, subzones, show_progress=False):
    cp_hcube_ref = cp.array(hcube_ref)
    cp_coeffs = cp.array(coeffs)
    cp_subzones = cp.array(subzones)
    _, _, ny, nx = cp_hcube_ref.shape # Number of: images, wavelengths, y-pixels, and x-pixels
    nL, nR, nT, _ = cp_coeffs.shape
    hcube_psfmodel = cp.zeros((nT, nL, ny, nx))+np.nan # Array in which to place reconstructed model
    for Ri in tqdm(range(nR), desc='PSF model reconstruction (regions)', leave=False) if show_progress else range(nR):
        sub_i = cp_subzones[Ri] # An ny by nx boolean mask indicating which pixels are being considered
        imvals = cp_hcube_ref[:, :, sub_i].transpose((1,0,2)) # Fetching the pixels in the subzone, dimensions of nT_ref x nL x npx
        for Ti in tqdm(range(nT), desc='PSF model reconstruction (exposures)', leave=False) if show_progress else range(nT): # Iterate over target images, building the PSF model (in the subzone) for each.
            cvals = cp_coeffs[:, Ri, Ti, :, cp.newaxis] # Fetch the appropriate coefficients
            hcube_psfmodel[Ti, :, sub_i] = cp.sum(cvals*imvals, axis=1) # Multiply images by the coefficients, then sum along the nT axis to get the model values.
    hcube_psfmodel_np = cp.asnumpy(hcube_psfmodel) # Convert output back to numpy
    hcube_psfmodel, cp_hcube_ref, cp_coeffs, cp_subzones, cvals = free_gpu(hcube_psfmodel, cp_hcube_ref, cp_coeffs, cp_subzones, cvals) # Explicitly clear VRAM for cupy arrays
    return hcube_psfmodel_np


def median_combine_sequence(hcube, use_gpu=False, axis=0):
    if use_gpu:
        out = numpy_to_gpu_nanmedian(hcube, axis=axis)
    else:
        out = np.nanmedian(hcube, axis=axis)
    return out


def numpy_to_gpu_nanmedian(x, axis=None):
    x_cp = cp.array(x)
    med_cp = gpu_nanmedian(x_cp, axis=axis)
    
    med_numpy = cp.asnumpy(med_cp)
    x_cp, med_cp = free_gpu(x_cp, med_cp)
    return med_numpy


def index_axis(index, axis):
    return (slice(None), )*axis + (index, )


def gpu_nanmedian(x, axis=None):
    """
    The cupy nanmedian operation can use a huge amount of VRAM.
    This breaks the nanmedian process into portions that should 
    be manageable with your available VRAM.
    """
    axes = np.arange(x.ndim)
    preserved_axes = axes[~np.isin(axes, np.asarray(axis))]
    
    in_shape = np.asarray(x.shape)
    out_shape = in_shape[preserved_axes]
    out = cp.zeros(tuple(out_shape), dtype=x.dtype)
    avail_mem = gpu.mem_info[0]+(cp.get_default_memory_pool().total_bytes() - cp.get_default_memory_pool().used_bytes())
    if avail_mem < (x.nbytes*6):
        nparts = int(np.ceil(x.nbytes*6 / avail_mem))
        sorted_axes = np.argsort(in_shape)
        split_axis = sorted_axes[np.isin(sorted_axes, preserved_axes)][-1]
        nsplit = in_shape[split_axis]
        split_inds = cp.array_split(cp.arange(nsplit), nparts)
        for inds in split_inds:
            out[index_axis(inds, np.where(preserved_axes == split_axis)[0][0])] = cp.nanmedian(x[index_axis(inds, split_axis)], axis=axis)
            free_gpu()
    else:
        cp.nanmedian(x, axis=axis, out=out)
        free_gpu()
    return out


def build_annular_rdi_zones(nx, ny, cent, r_opt=None, r_sub=None, pxscale=None, distance=None):
    """
    Generates a set of optimization and subtraction zones for PSF-subtraction.
    
    ___________
    Parameters:

    nx: int
        x-axis dimension of the data for which to create zones

    ny: int
        y-axis dimension of the data for which to create zones
        
    cent: ndarray
        length two array giving the center coordinate for the zones
        as [x_center, y_center]
    _________
    Optional:

    r_opt: array, float, or None
        The optimization zone radii. 
        
        If array: either a) a two dimensional array where each entry along
            the leading axis gives the inner and outer radius for an
            optimization zone (e.g., r_opt = [[r1_zone1, r2_zone1],
            [r1_zone2, r2_zone2]]), or b) a one dimensional array giving
            the inner and outer radii for a single optimization zone 
            (e.g., r_opt = [r1, r2])
            
        If float: assume r_opt provides the outer radius for a single zone
            with inner radius of 0
            
        If r_opt has Astropy units: 
            if units are an angle (e.g., arcsec, mas), use pxscale to convert to 
            pixels; if units are a physical length (e.g., au) use distance and 
            pxscale to convert to pixels
        
        If None: a single optimization zone spanning the entire FOV is used
        
    r_sub: array, float, or None
        The subtraction zone radii. Should match the shape of r_opt unless None.
        Options as for r_opt, except if None, where subtraction zones are generated
        for the same radii as the optimization zones, except that the inner radius
        for the innermost zone is set to zero, and the outer radius for the outermost
        zone extends to the edge of the field of view. E.g.,
        r_opt=[[5,15],[15,50]] and r_sub=None will use r_sub=[[0,15],[15,np.inf]]
        
    pxscale: float, astropy.units.quantity.Quantity or None
        Optional only if r_opt / r_sub do not have astropy units of angle or length. 
        The pixel scale for the data. If a unitless float, value is assumed to be
        arcsec/pixel. If having astropy units, the units must be equivalent to 
        arcsec/pixel (e.g., mas/pixel, deg/pixel).
        
    distance: float, astropy.units.quantity.Quantity or None
        Optional only if r_opt / r_sub do not have astropy units of length. 
        The distance to the target star. If a unitless float, value is assumed to be
        parsec. If having astropy units, the units must be equivalent to 
        parsecs (e.g., lightyears).
        
    _________
    Returns:
    
        optzones, subzones:
            Each a numpy boolean array of shape (nZ, ny, nx) where nZ is the specified
            number of zones.
    """
    rmap = dist_to_pt(cent, nx, ny)
    
    if isinstance(r_opt, u.quantity.Quantity):
        r_opt, r_opt_unit = r_opt.value, r_opt.unit
    else:
        r_opt_unit = 1
        
    if isinstance(r_sub, u.quantity.Quantity):
        r_sub, r_sub_unit = r_sub.value, r_sub.unit
    else:
        r_sub_unit = r_opt_unit
    
    if r_opt is None:
        r_opt = [[0, np.inf]]
    elif np.ndim(r_opt) == 0: # If a single value
        r_opt = [[0, r_opt]]
    elif np.ndim(r_opt) == 1: # If a single lower and upper limit
        r_opt = [r_opt]
        
    if r_sub is None:
        r_sub = deepcopy(r_opt)
        r_sub[0][0] = 0
        r_sub[-1][1] = np.inf
    elif np.ndim(r_sub) == 0: # If a single value
        r_sub = [[0, r_sub]]
    elif np.ndim(r_sub) == 1: # If a single lower and upper limit
        r_sub = [r_sub]

    r_opt = r_opt * r_opt_unit
    r_sub = r_sub * r_sub_unit

    if isinstance(r_opt, u.quantity.Quantity):
        if r_opt.unit.is_equivalent(u.au):
            if distance is None:
                raise ValueError("""If r_opt / r_sub are provided with units of length,
                distance must be provided.""")
            r_opt = proj_sep_to_ang_size(r_opt, distance)
        if r_opt.unit.is_equivalent(u.arcsec):
            if pxscale is None:
                raise ValueError("""If r_opt / r_sub are provided with angle or length units,
                pxscale must be provided.""")
            r_opt = ang_size_to_px_size(r_opt, pxscale)
        r_opt = r_opt.value
        
    if isinstance(r_sub, u.quantity.Quantity):
        if r_sub.unit.is_equivalent(u.au):
            if distance is None:
                raise ValueError("""If r_opt / r_sub are provided with units of length,
                distance must be provided.""")
            r_sub = proj_sep_to_ang_size(r_sub, distance)
        if r_sub.unit.is_equivalent(u.arcsec):
            if pxscale is None:
                raise ValueError("""If r_opt / r_sub are provided with angle or length units,
                pxscale must be provided.""")
            r_sub = ang_size_to_px_size(r_sub, pxscale)
        r_sub = r_sub.value
        
    optzones = []
    subzones = []
    
    for (ro,rs) in zip(r_opt, r_sub):
        optzones.append((rmap >= ro[0])&(rmap < ro[1]))
        subzones.append((rmap >= rs[0])&(rmap < rs[1]))
        
    optzones, subzones = np.asarray(optzones), np.asarray(subzones)

    # Add any pixels not included in any subtraction zone to the last one
    no_sub_coverage = np.all(~subzones, axis=0)
    subzones[-1] = np.logical_or(subzones[-1], no_sub_coverage)
    return (optzones, subzones)

try:
    import cupy as cp
    gpu = cp.cuda.Device(0)
except ModuleNotFoundError:
    ""