import numpy as np
import astropy.units as u
import torch
from copy import copy, deepcopy
from tqdm.auto import tqdm

from .utils import (dist_to_pt, rotate_hypercube, pad_and_rotate_hypercube, ang_size_to_px_size, proj_sep_to_ang_size)
from .backend import get_device, resolve_dtype, to_tensor, to_numpy

def rdi_residuals(hcube, hcube_ref, optzones, subzones, hcube_css=None, ref_mask=None, prepared_refmats=None, show_progress=False, posangs=None, cent=None, 
                  objective=False, zero_nans=False, device=None, return_coeffs=False, coeffs_in=None, return_psf_model=False,
                  pad_before_derot=False, opt_smoothing_fn=None, opt_smoothing_kwargs={}, err_hcube=None, err_hcube_ref=None):
    """
    Performs RDI PSF subtraction using LOCI (Lafreniere et al. 2007). Using available keyword arguments, this function can be used to 
    perform:

    a) Model Constrained RDI (MCRDI; Lawson et al. 2022) — by passing in a convolved model of the circumstellar scene (CSS) via hcube_css 
    b) Polarimetry Constrained RDI (PCRDI; Lawson et al. 2022) — by passing a polarimetry-based total intensity estimate of the CSS via
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
            4D array; same shape as hcube. hcube_css should provide an estimate of the circumstellar scene (CSS)
            in hcube, rotated to the appropriate posangs and convolved with the appropriate PSF.
            
        ref_mask: ndarray
            2D boolean array of shape (len(optzones), nT_ref) that indicates which reference images should be considered
            for which optimization regions. E.g., if ref_mask[i,j] is False, then for the ith optimization zone (optzones[i]),
            the jth reference image (hcube_ref[j]) will NOT be used for construction of the PSF model. This can be useful if
            some reference exposures have anomalous features that make them problematic for some regions while still being
            suitable for others; e.g., an image with a bright background source near the edge of the FOV may still be useful
            for nulling the PSF near the inner working angle.

        prepared_refmats: dict or None
            Dictionary containing precomputed QR decompositions of the reference images. Should have the format:
            {'Qs': [Q1, Q2, ...], 'Rs': [R1, R2, ...]}, where each Qi and Ri correspond to the QR decomposition
            of the ith reference image block. If provided, these will be used instead of computing the QR decomposition
            from scratch. See _prepare_reference_matrices for default implementation.
            
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
            
        device: str, torch.device, or None
            The device on which to carry out the calculations (e.g., 'cpu', 'mps', 'cuda'). If None, the default device
            is used (see winnie.backend). Calculations on devices that do not support float64 (e.g., 'mps') are carried
            out in float32.
            
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
        hcube_opt = opt_smoothing_fn(hcube_sub, **opt_smoothing_kwargs)
        if (prepared_refmats is None) or objective: # only create the filtered optimization refs when not passing in prepared_refmats
            hcube_ref_opt = opt_smoothing_fn(hcube_ref, **opt_smoothing_kwargs)
        else:
            hcube_ref_opt = hcube_ref
    else:
        hcube_opt = hcube_sub
        hcube_ref_opt = hcube_ref

    if zero_nans: 
        hcube_opt = np.nan_to_num(hcube_opt)
        hcube_ref_opt = np.nan_to_num(hcube_ref_opt)
    
    if coeffs_in is None:
        coeffs = compute_rdi_coefficients(hcube_opt, hcube_ref_opt, optzones, show_progress=show_progress, ref_mask=ref_mask, prepared_refmats=prepared_refmats, device=device)
    else:
        coeffs = coeffs_in

    if return_coeffs:
        return coeffs

    if objective:
        psf_model, err_psf_model = reconstruct_psf_model(hcube_ref_opt, coeffs, subzones, show_progress=show_progress, device=device, err_hcube_ref=err_hcube_ref)
    else:
        psf_model, err_psf_model = reconstruct_psf_model(hcube_ref, coeffs, subzones, show_progress=show_progress, device=device, err_hcube_ref=err_hcube_ref)

    if return_psf_model:
        return psf_model
    
    if objective:
        hcube_res = hcube_opt - psf_model
    else:
        hcube_res = hcube - psf_model

    cent_out = copy(cent)
    if posangs is not None:
        if pad_before_derot:
            hcube_res, cent_out = pad_and_rotate_hypercube(hcube_res, -posangs, cent=cent, device=device, cval0=np.nan)
        else:
            hcube_res = rotate_hypercube(hcube_res, -posangs, cent=cent, device=device, cval0=np.nan)

    if err_hcube is None:
        err_hcube_res = None
    else:
        err_hcube_res = np.abs(err_hcube) if err_psf_model is None else np.hypot(err_hcube, err_psf_model)
        if posangs is not None:
            if pad_before_derot:
                err_hcube_res, _ = pad_and_rotate_hypercube(err_hcube_res, -posangs, cent=cent, device=device, cval0=np.nan)
            else:
                err_hcube_res = rotate_hypercube(err_hcube_res, -posangs, cent=cent, device=device, cval0=np.nan)
    return hcube_res, err_hcube_res, cent_out


def _prepare_reference_matrices(hcube_ref, optzones, ref_mask=None):
    """
    Computes the QR decomposition of the reference images for each optimization zone (and wavelength), in the dtype of
    hcube_ref. These do not depend on the science images, so they can be cached and re-used (see prepared_refmats in
    rdi_residuals). This is always done on the CPU as it only needs to be done once and torch.linalg.qr is not reliable
    for large matrices on 'mps'.
    """
    prepared_refmats = dict(Qs=[], Rs=[])
    for Ri in range(len(optzones)):
        opt_i = optzones[Ri]
        refarrs = hcube_ref[..., opt_i].transpose(1, 2, 0)
        if ref_mask is not None:
            refarrs = refarrs[..., ref_mask[Ri]]
        Q, R = torch.linalg.qr(to_tensor(refarrs, 'cpu'), mode='reduced')
        prepared_refmats['Qs'].append(to_numpy(Q))
        prepared_refmats['Rs'].append(to_numpy(R))
    return prepared_refmats


def compute_rdi_coefficients(hcube, hcube_ref, optzones, show_progress=False, ref_mask=None, prepared_refmats=None, device=None):
    device = get_device(device)
    nR = len(optzones)
    coeff_hcube = np.zeros((nR, hcube.shape[0], hcube_ref.shape[0], hcube.shape[1]), dtype=hcube_ref.dtype)

    if prepared_refmats is None:
        prepared_refmats = _prepare_reference_matrices(hcube_ref, optzones, ref_mask=ref_mask)
    dtype = resolve_dtype(device, hcube.dtype, hcube_ref.dtype, prepared_refmats['Qs'][0].dtype)

    Riterator = (tqdm(range(nR), desc='Coefficient calculation (regions)', leave=False) if show_progress
                 else range(nR))

    for Ri in Riterator:
        tararrs = to_tensor(hcube[..., optzones[Ri]], device, dtype).permute(1, 2, 0)
        Q, R = to_tensor(prepared_refmats['Qs'][Ri], device, dtype), to_tensor(prepared_refmats['Rs'][Ri], device, dtype)
        coeffs = to_numpy(torch.linalg.solve_triangular(R, Q.mT @ tararrs, upper=True).permute(2, 1, 0))
        if ref_mask is not None:
            coeff_hcube[Ri][:, ref_mask[Ri], :] = coeffs
        else:
            coeff_hcube[Ri] = coeffs
    return coeff_hcube


def reconstruct_psf_model(hcube_ref, coeffs, subzones, show_progress=False, device=None, err_hcube_ref=None):
    """
    Reconstructs the PSF model for each science exposure by linearly combining the reference images with the
    coefficients (with shape (nR, nT, nT_ref, nL)) over each subtraction zone. This is done as a matrix product on
    device (rather than by forming the product of the coefficients with every reference image), so the memory
    required is limited to that of the inputs and outputs for the zone being processed.

    If err_hcube_ref is provided, the uncertainties in the model are propagated from it (assuming uncorrelated
    uncertainties). Returns the PSF model and its uncertainties (or None) as numpy arrays.
    """
    device = get_device(device)
    nR, nT, _, nL = coeffs.shape # regions, science images, reference images, wavelengths
    ny, nx = hcube_ref.shape[-2:]
    dtype = resolve_dtype(device, hcube_ref.dtype, coeffs.dtype)
    hcube_ref_t, coeffs_t = to_tensor(hcube_ref, device, dtype), to_tensor(coeffs, device, dtype)
    hcube_psfmodel = torch.full((nT, nL, ny, nx), float('nan'), dtype=dtype, device=device) # Array in which to place reconstructed model

    calc_err = err_hcube_ref is not None
    if calc_err:
        err_hcube_ref_t = to_tensor(err_hcube_ref, device, dtype)
        err_hcube_psfmodel = torch.full_like(hcube_psfmodel, float('nan'))

    for Ri in tqdm(range(nR), desc='PSF model reconstruction (regions)', leave=False) if show_progress else range(nR): # Iterate over regions
        sub_i = to_tensor(subzones[Ri], device) # An ny by nx boolean mask indicating which pixels are being considered
        c_i = coeffs_t[Ri] # The coefficients for this region, dimensions of nT x nT_ref x nL
        # Sum over the reference images (r) of the coefficients times the subzone pixels (nT_ref x nL x npx) for each science image (t) and wavelength (l)
        hcube_psfmodel[..., sub_i] = torch.einsum('trl,rlp->tlp', c_i, hcube_ref_t[:, :, sub_i])
        if calc_err:
            err_hcube_psfmodel[..., sub_i] = torch.sqrt(torch.einsum('trl,rlp->tlp', c_i**2, err_hcube_ref_t[:, :, sub_i]**2))

    return (to_numpy(hcube_psfmodel, hcube_ref.dtype),
            to_numpy(err_hcube_psfmodel, hcube_ref.dtype) if calc_err else None)


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
        r_sub[-1][1] = int(np.ceil(np.max(rmap)) + 1)
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