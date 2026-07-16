from .utils import (xy_polar_ang_displacement, c_to_c_osamp, pad_or_crop_image, px_size_to_ang_size, ang_size_to_proj_sep, ang_size_to_px_size, dist_to_pt)
from .distortion import get_siaf_aper
from .convolution import psf_convolve_cpu
from .rdi import (compute_rdi_coefficients)

import numpy as np
import astropy.units as u
from copy import deepcopy
import lmfit
from astropy.io import fits
from webbpsf_ext import image_manip
from astropy.convolution import Gaussian2DKernel
from vip_hci.fm import ScatteredLightDisk
from IPython.display import clear_output

try:
    import stpsf
except ModuleNotFoundError:
    import webbpsf as stpsf
    

_RING_PARAM_DEFAULTS = dict(pa=[90., -360., 360.], # -360 to +360 to avoid discontinuities for optimizers
                            incl=[30., -89.5, 89.5],
                            ain=[10., 1.1, 100],
                            aout=[-10., -100., -1.1],
                            h0=[0.05, 0.001, 0.1],
                            beta=[1., 0., 2.],
                            gamma=[2., 0.1, 6.],
                            e=[0., 0., 0.9],
                            omega=[0.,-90.,90.])


def lmfit_nrings_nptsrc_init(ptsrc_params=[], ring_params=[], vary_beta=False, vary_gamma=False, vary_e=False, coplanar_rings=True, coeccentric_rings=True, spf='twohg',
                             ptsrc_dxy_range=0.5, ptsrc_det_blur=None, ptsrc_sky_blur=None, vary_dist_coeffs=False, siaf_aper=None, autoscale_all=True):
    """
    Initializes an LMFIT parameters object for a circumstellar scene composed of N_rings rings and N_ptsrc point sources. 
    
    ptsrc_params is a list where each entry is a dictionary containing at least 'dx' and 'dy' (in arcsec) of a point-like source relative to the star, 
    with an optional 3rd entry, 'flux', to initialize the flux (in mJy) of the source. Coordinates should be in the North-up frame with positive 'dx'
    to the west of the star, and positive 'dy' to the north of the star.
    
    ring_params is a list where each entry is a dictionary containing at least 'r0', the fiducial radius (in au) of the ring. If a dictionary key is
    array-like with three entries, the values are assumed to be [initial, min, max]. Other valid entries are 'pa', 'incl', 'ain', 'aout', 'h0', 'beta',
    'gamma', 'e', and 'omega'. Note: if coplanar_rings is True, the PA and incl input for any rings after the first will be ignored. Likewise for 
    coeccentric_rings and e / omega. 
    
    vary_beta, vary_gamma, and vary_e dictate whether these optional parameters are set to vary (all False by default; vary_e also sets omega to vary).
    
    coplanar_rings: if True, all rings are required to have the same inclination and PA. Otherwise, each ring's PA and inclination vary separately.
    
    coeccentric_rings: if True, all rings have the same eccentricity and argument of periapsis. 

    spf: either 'twohg', 'onehg', or 'isotropic'
        'twohg': the scattering phase function (SPF) for each ring is described by the linear combination (weight varying) of two Henyey-Greenstein (HG) SPFs, 
                 each with a varying asymmetry parameter (g1 and g2).
        'onehg': the scattering phase function (SPF) for each ring is described by a single HG SPF with a single varying parameter, g1.
        'isotropic': Isotropic scattering is emulated by fixing the asymmetry parameter for a single HG SPF to 0.
    
    ptsrc_dxy_range: the maximum offset of each point source from the initial position provided.
    
    ptsrc_det_blur: either None, 'gaussian', or 'gaussian2d'. If None, point sources get no additional blurring. 
                    If 'gaussian', point sources are blurred using a single-parameter 2D gaussian (equal sigma in x and y range). 
                    If 'gaussian2d', point sources are blurred with a different x and y sigma, with angle theta relative to the detector frame — i.e., 
                    suitable for emulating extra detector effects for sources that are truly point-like.
                    
    ptsrc_sky_blur: either None, 'gaussian', or 'gaussian2d'. If None, point sources get no additional blurring. 
                    If 'gaussian', point sources are blurred using a single-parameter 2D gaussian (equal sigma in x and y range). 
                    If 'gaussian2d', point sources are blurred with a different x and y sigma, with angle theta relative to the sky frame — i.e., 
                    suitable for emulating slightly extended sources (e.g., marginally resolved background galaxies)
                    
    vary_dist_coeffs: if True, will include a subset of astrometric distortion parameters, initialized based on pysiaf values. If a list of coefficient names
                      (e.g., "Sci2IdlX10"), only the included entries will be set to vary.
    
    siaf_aper: a PySIAF aperture object corresponding to the observations being fit (can use wdb.siaf_aper, or prepare one separately with 
               winnie.distortion.get_siaf_aper()). Only used if vary_dist_coeffs is not False.
                        
    """
    if ptsrc_det_blur not in [None, 'gaussian', 'gaussian2d']:
        raise ValueError("ptsrc_det_blur must be None, 'gaussian', or 'gaussian2d'")
        
    if ptsrc_sky_blur not in [None, 'gaussian', 'gaussian2d']:
        raise ValueError("ptsrc_sky_blur must be None, 'gaussian', or 'gaussian2d'")
    
    init_params = lmfit.Parameters()
    
    #########################
    # Disk / ring components:
    #########################
    
    vary_ring_pars = dict(r0=True, 
                     pa=True,
                     incl=True,
                     ain=True,
                     aout=True,
                     h0=True,
                     beta=vary_beta,
                     gamma=vary_gamma,
                     e=vary_e,
                     omega=vary_e)
    
    for i, entry in enumerate(ring_params):
        if 'r0' not in entry:
            raise ValueError("Each dictionary of ring parameters must contain at least 'r0'.")
        
        if np.size(entry['r0']) == 3:
            v0, vmin, vmax = entry['r0']
        else:
            v0, vmin, vmax = entry['r0'], 1., 1000.
         
        init_params.add(f'r0_{i+1}', value=v0, min=vmin, max=vmax)
        
        for key in _RING_PARAM_DEFAULTS:
            if (i>0) and np.any([(key in ['pa', 'incl']) and coplanar_rings,
                                 (key in ['e','omega']) and coeccentric_rings]):
                init_params.add(f'{key}_{i+1}', expr=f'{key}_1')
                
            else:
                if key not in entry:
                    v0, vmin, vmax = _RING_PARAM_DEFAULTS[key]
                elif np.size(entry[key]) == 3:
                    v0, vmin, vmax = entry[key]
                else:
                    v0, vmin, vmax = entry[key], _RING_PARAM_DEFAULTS[key][1], _RING_PARAM_DEFAULTS[key][2]
                    
                init_params.add(f'{key}_{i+1}', value=v0, min=vmin, max=vmax, vary=vary_ring_pars[key])
            
        # Scattering parameters
        if spf == 'isotropic':
            init_params.add(f'g1_{i+1}',   value=0.0, min=-0.99, max=0.99, vary=False)
        else:
            init_params.add(f'g1_{i+1}',   value=0.8, min=0., max=0.99, vary=True)
    
        if spf == 'twohg':
            init_params.add(f'g2_{i+1}',  value=-0.2, min=-0.99, max=0.99, vary=True)
            init_params.add(f'wg1_{i+1}', value=0.65, min=0.0, max=1.0, vary=True)
        else:
            init_params.add(f'g2_{i+1}',  value=0.0, min=-0.99, max=0.99, vary=False)
            init_params.add(f'wg1_{i+1}', value=1.0, min=0.0,   max=1.0,  vary=False)  
        
        init_params.add(f'F_{i+1}', value=1, min=0.01, max=100, vary=(False if autoscale_all else (i>0)))
           
    ################
    # Point sources:
    ################
    
    for i, entry in enumerate(ptsrc_params):
        if ('dx' not in entry) or ('dy' not in entry):
            raise ValueError("Each dictionary of point source parameters must contain at least 'dx' and 'dy'.")
            
        init_params.add(f'ptsrc_dx_{i+1}',   value=entry['dx'], min=entry['dx']-ptsrc_dxy_range, max=entry['dx']+ptsrc_dxy_range, vary=True)
        init_params.add(f'ptsrc_dy_{i+1}',   value=entry['dy'], min=entry['dy']-ptsrc_dxy_range, max=entry['dy']+ptsrc_dxy_range, vary=True)
        init_params.add(f'ptsrc_flux_{i+1}', value=(1 if (('flux' not in entry) or autoscale_all) else entry['flux']), min=0., max=np.inf, vary=(not autoscale_all))
        
        if ptsrc_det_blur is not None:
            init_params.add(f'ptsrc_detxsig_{i+1}', value=1e-10, min=1e-10, max=10, vary=True)
            if ptsrc_det_blur == 'gaussian':
                init_params.add(f'ptsrc_detysig_{i+1}', expr=f'ptsrc_detxsig_{i+1}')
                init_params.add(f'ptsrc_dettheta_{i+1}', value=0.0, vary=False)
            else:
                init_params.add(f'ptsrc_detysig_{i+1}', value=1e-10, min=1e-10, max=10, vary=True)
                init_params.add(f'ptsrc_dettheta_{i+1}', value=0.0, min=-180, max=180, vary=True)
                
        if ptsrc_sky_blur is not None:
            init_params.add(f'ptsrc_skyxsig_{i+1}', value=1e-10, min=1e-10, max=10, vary=True)
            if ptsrc_sky_blur == 'gaussian':
                init_params.add(f'ptsrc_skyysig_{i+1}', expr=f'ptsrc_skyxsig_{i+1}')
                init_params.add(f'ptsrc_skytheta_{i+1}', value=0.0, vary=False)
            else:
                init_params.add(f'ptsrc_skyysig_{i+1}', value=1e-10, min=1e-10, max=10, vary=True)
                init_params.add(f'ptsrc_skytheta_{i+1}', value=0.0, min=-180, max=180, vary=True)
    
    #########################
    # Astrometric distortion:
    #########################
    
    if isinstance(vary_dist_coeffs, bool):
        if vary_dist_coeffs:
            varying_dist_coeffs=['Sci2IdlX10', 'Sci2IdlY10', 'Sci2IdlY11']
        else:
            varying_dist_coeffs=[]

    else: # assume its a list of coeff names
        varying_dist_coeffs = vary_dist_coeffs
        
    if len(varying_dist_coeffs) != 0:
        if siaf_aper is None:
            raise ValueError("If vary_dist_coeffs is not False, you must also provide 'siaf_aper', a PySIAF aperture object for your observing aperture!")

        for par in varying_dist_coeffs:
            if siaf_aper.__dict__[par] == 0:
                print(f'Warning: {par} is zero in PySIAF! Varying this parameter may produce problematic results!')
            if par in ['Sci2IdlX10', 'Sci2IdlY11']:
                init_params.add(par, value=siaf_aper.__dict__[par], vary=True, min=siaf_aper.__dict__[par]*0.95, max=siaf_aper.__dict__[par]*1.05)
            else:
                init_params.add(par, value=siaf_aper.__dict__[par], vary=True, min=-0.1, max=0.1)
                
    return init_params


def get_ptsrc_model_coords(dxy_northup, posang, xy_star_sci, xy_mask_sci, siaf_aper, pxscale=None, distorted_coords=False):
    xy_mask_sci0 = np.array([siaf_aper.XSciRef, siaf_aper.YSciRef])-1

    if pxscale is None:
        pxscale = 0.5*(siaf_aper.XSciScale+siaf_aper.YSciScale)*u.arcsec/u.pixel

    dxy = np.asarray(xy_polar_ang_displacement(*dxy_northup, -posang))

    if distorted_coords:
        xy_sci = dxy / pxscale.value + xy_star_sci
        dxy_coron = siaf_aper.sci_to_idl(*(xy_sci - xy_mask_sci + xy_mask_sci0 + 1))

    else:
        xy_sci = siaf_aper.idl_to_sci(*dxy) - xy_mask_sci0 + xy_mask_sci - 1
        dxy_coron = dxy + siaf_aper.sci_to_idl(*(xy_star_sci-xy_mask_sci+xy_mask_sci0+1))

    return np.array(xy_sci), np.array(dxy_coron)


def generate_ptsrc_hduls(spacerdi, ptsrc_params, spectrum=None, fov_pixels=151, osamp=2, distorted_coords=True, nlambda=None, normalize='exit_pupil'):
    """
    ptsrc_params: a list of dictionaries, each containing at least 'dx' and
        'dy' (in arcsec) of a point-like source relative to the star, with an
        optional 3rd entry, 'flux', to initialize the flux (in mJy) of the source.
        Coordinates should be in the North-up frame with positive 'dx' to the west
        of the star, and positive 'dy' to the north of the star. The flux is
        assumed to be 1 mJy if not provided.
    """
    if spectrum is None:
        spectrum = stpsf.specFromSpectralType('M5V', catalog='ck04') # Latest spectral type in this library
    if spacerdi.convolver.inst_stpsf is None:
        spacerdi.convolver.prepare_stpsf_instance(spacerdi.convolver.stpsf_options)
        
    inst = spacerdi.convolver.inst_stpsf
    siaf_aper = inst.siaf[inst.aperturename]
    options = deepcopy(inst.options)
    ptsrc_hduls = []
    
    charge_diffusion_options = dict(charge_diffusion_sigma=inst.options.get('charge_diffusion_sigma', None))

    for j, posang in enumerate(spacerdi._posangs_sci):
        c_coron = spacerdi.c_coron_sci[j]
        ptsrc_hduls_roll = []
        for i,ptsrc_params_i in enumerate(ptsrc_params):
            dxy_northup = np.array([ptsrc_params_i['dx'], 
                                    ptsrc_params_i['dy']])
                
            xy_sci, dxy_coron = get_ptsrc_model_coords(np.asarray(dxy_northup), posang, spacerdi.c_star, c_coron, siaf_aper, pxscale=spacerdi.pxscale, distorted_coords=distorted_coords)
            det_pos = xy_sci - c_coron + (np.array([siaf_aper.XSciRef, siaf_aper.YSciRef])) # We don't subtract 1 from the ref pos because we need a 1-indexed position
            inst.detector_position = np.clip(det_pos, 0, np.array(inst._detector_npixels)-1)
            inst.options['coron_shift_x'], inst.options['coron_shift_y'] = -np.asarray(dxy_coron)
            source = (spectrum[i] if isinstance(spectrum, list) else spectrum)

            hdul = inst.calc_psf(source=source, fov_pixels=fov_pixels, oversample=osamp, nlambda=nlambda, normalize=normalize)
            
            hdul_nodist = stpsf.distortion.apply_rotation(fits.HDUList([hdul[0], hdul[0]]), crop=True)
            hdul[0] = stpsf.detectors.apply_detector_charge_diffusion(hdul_nodist, charge_diffusion_options)[1] # this is applied after astrometric distortion in STPSF, but the difference is small
        
            ptsrc_hduls_roll.append(hdul)

        ptsrc_hduls.append(ptsrc_hduls_roll)
        
    # Set inst back as it was
    inst.options = options
    inst.detector_position = (siaf_aper.XSciRef, siaf_aper.YSciRef)
    return ptsrc_hduls


def generate_nptsrc_model_cube(spacerdi, ptsrc_params, ptsrc_hduls=None, blur_params={}, dist_params={}, npad=0, return_components=False, distorted_coords=True, siaf_aper=None, apply_tcoron=True, ext=2, **gen_kwargs):
    """
    ptsrc_params: a list of dictionaries, each containing at least 'dx' and
        'dy' (in arcsec) of a point-like source relative to the star, with an
        optional 3rd entry, 'flux', to initialize the flux (in mJy) of the source.
        Coordinates should be in the North-up frame with positive 'dx' to the west
        of the star, and positive 'dy' to the north of the star. The flux is
        assumed to be 1 mJy if not provided.
    """
    if ptsrc_hduls is None:
        ptsrc_hduls = generate_ptsrc_hduls(spacerdi, ptsrc_params, **gen_kwargs)
    if spacerdi.convolver._psf_shift is None:
        if spacerdi.convolver.inst_webbpsfext is None:
            spacerdi.convolver.prepare_webbpsf_ext_instance(spacerdi.convolver.stpsf_options)
        spacerdi.convolver.calc_psf_shift()

    if siaf_aper is None:
        siaf_aper = get_siaf_aper(spacerdi.convolver.aperturename, spacerdi.convolver.instrument, dist_params)
        
    nr, ny, nx = spacerdi.imcube_sci.shape
    
    hcube = np.zeros((len(ptsrc_hduls[0]), nr, ny+2*npad, nx+2*npad), dtype=spacerdi.imcube_sci.dtype)

    c_star = spacerdi.c_star+npad
    for j, posang in enumerate(spacerdi._posangs_sci):
        c_coron = spacerdi.c_coron_sci[j]+npad
        for i, hdul in enumerate(ptsrc_hduls[j]):
            osamp = hdul[2].header['OVERSAMP']

            ptsrc_params_i = ptsrc_params[i]
            flux = ptsrc_params_i.get('flux', 1.0)
            dxy_northup = np.array([ptsrc_params_i['dx'], 
                                    ptsrc_params_i['dy']])
            
            xy_sci, dxy_coron = get_ptsrc_model_coords(dxy_northup, posang, c_star, c_coron, siaf_aper, pxscale=spacerdi.pxscale, distorted_coords=distorted_coords)
            
            if apply_tcoron:
                dxy_coron_jitter = dxy_coron[:, np.newaxis] + np.random.normal(scale=spacerdi.convolver.inst_stpsf.options['jitter_sigma'], size=(2,1000))
                t_coron = np.mean(spacerdi.convolver.inst_webbpsfext.gen_mask_transmission_map(dxy_coron_jitter, coord_frame='idl'))
            else:
                t_coron = 1
                
            psf = hdul[ext].data.copy()

            psf = psf * flux * t_coron

            if f'detxsig_{i+1}' in blur_params:
                det_kernel = Gaussian2DKernel(blur_params[f'detxsig_{i+1}']*osamp, blur_params[f'detysig_{i+1}']*osamp, np.deg2rad(blur_params[f'dettheta_{i+1}']))
                psf = psf_convolve_cpu(psf, det_kernel)

            if f'skyxsig_{i+1}' in blur_params:
                sky_kernel = Gaussian2DKernel(blur_params[f'skyxsig_{i+1}']*osamp, blur_params[f'skyysig_{i+1}']*osamp, np.deg2rad(blur_params[f'skytheta_{i+1}']-posang))
                psf = psf_convolve_cpu(psf, sky_kernel)
                
            if ext==0:
                hdul_in = fits.HDUList([fits.PrimaryHDU(psf, hdul[ext].header)])
                psf = stpsf.distortion.distort_image(hdul_in, ext=0, aper=siaf_aper)

            im = pad_or_crop_image(psf, np.array(hcube.shape[-2:])*osamp,
                           cent=None, new_cent=c_to_c_osamp(xy_sci+spacerdi.convolver._psf_shift, osamp), cval0=0.,
                           nan_prop_threshold=1e-8, zero_prop_threshold=1e-8)
            
            if osamp != 1:
                im = image_manip.frebin(im, scale=1./osamp, total=True)
            hcube[i,j] = im

    hcube = (hcube*(u.mJy/u.pixel**2)/spacerdi.pxscale**2).to(u.MJy/u.sr).value
    
    if return_components:
        return hcube
    return np.nansum(hcube, axis=0)


counter=0

def obj_fn(p, rdi_reduc, wdb, roi, distance, ptsrc_hduls=None, err_weighting=False, rmax_accuracy=None,
           halfNbSlices=25, return_soln=False, q_clip=None, modelcubes_in=[], modellabels_in=[], fmreducs_in=[], autoscale_in=[],
           disk_model_osamp=1, ptsrc_gen_kwargs={}, distorted_coords=True, add_fm_eps=True, count=True, clear_each_call=True):
    """
    Parameters
    ----------
    p: lmfit.parameter.Parameters
        LMFit parameters object as initialized by lmfit_nrings_nptsrc_init()
        and updated by the optimizer.
        
    rdi_reduc: winnie.space.SpaceReduction
        An RDI reduction of the data using the current settings in 'wdb', the
        Winnie SpaceRDI object; the forward-modeled disk image will be compared
        to rdi_reduc.rolls
        
    wdb: winnie.space.SpaceRDI
        The Winnie SpaceRDI object that was used to generate rdi_reduc
        
    roi: numpy.ndarray
        Boolean 'region of interest' array having the same shape as
        rdi_reduc.im that indicates which pixels should be included in the
        goodness of fit evaluation.
        
    distance: float
        Distance to the target star in parsecs — used to generate disk models.
        
    ptsrc_hduls: list of list of astropy.io.fits.HDUList
        Contains one list for each roll, which contains one STPSF HDUList for
        each point source being modeled. If None, will generate the HDULists on
        the fly for any point source parameters in p. Leaving as None during
        optimization will add an enormous amount of runtime.
    
    err_weighting: bool
        If err_weighting is True, the array stored in rdi_reduc.err_rolls is
        used to weight the residuals. If an array of the same shape as
        rdi_reduc.rolls, then that array is used instead. If False, residuals
        are not weighted.
        
    rmax_accuracy: float
        The largest separation (in au) at which to calculate the disk model. If
        None, defaults to the edge of the FOV.
        
    halfNbSlices: int
        The number of planar slices to compute above and below the disk
        midplane when generating the raw disk model.
        
    return_soln: bool
        If True, rather than returning a residual array, returns the forward
        modeled image and some other items
        
    q_clip: tuple or list or numpy.ndarray
        If not None, q_clip gives a lower and upper quantile bound for the
        residuals. Any values outside the quantile range are clipped when
        evaluating goodness of fit. Can be useful for data with significant
        artifacts within the region of interest (e.g., poor reference match,
        uncorrected hot pixels, etc). E.g., q_clip = [5,95] will compute
        goodness of fit using only the inner 5th-95%ile of the distribution of
        residual pixel values in your region of interest.
    
    modelcubes_in: list
        Convolved model cubes with shape matching wdb.imcube_sci. Each entry
        corresponds to a different component of the model not being explicitly
        varied (e.g., the 'softwarebg' term in the Tutorial 6 notebook)
    
    modellabels_in: list
        String labels corresponding to the entries in modelcubes_in.

    fmreducs_in: list
        Forward modeled reductions of the entries in modelcubes_in. 

    autoscale_in: list
        Whether or not to automatically rescale the brightness of the
        corresponding component in modelcubes_in/fmreducs_in. 
        
    disk_model_osamp: int
        Oversampling factor for the raw disk model. Values > 1 will increase
        runtimes but may be necessary for especially compact and/or narrow disks.

    ptsrc_gen_kwargs: dict
        Additional keyword arguments to pass to generate_ptsrc_hduls() if point
        sources are being modeled and ptsrc_hduls is None.

    distorted_coords: bool
        If True, any point source positions are interpreted as distorted (i.e.,
        north-up stellocentric offsets naively assuming a uniform pixelscale of
        wdb.pxscale). If False, positions are interpreted as undistorted (i.e.,
        'idl' coordinates in SIAF terms).

    add_fm_eps: bool
        If True, adds a negligible random noise term to each image being forward
        modeled. Avoids a crash in the RDI coefficient calculation when the
        model is exactly zero across the entire optimization zone.

    count: bool
        If True, will advance and display a counter during the optimization.
        Requires setting 'counter = 0' somewhere outside of the function.
    
    clear_each_call: bool
        If True, clears the output every time the function is called. Mostly to
        prevent accumulating a large number of print statements if the loop
        triggers a warning somewhere.
    """

    pdict = p.valuesdict()

    global pdict_current
    pdict_current = deepcopy(pdict)

    if isinstance(err_weighting, bool):
        if err_weighting:
            sig = rdi_reduc.err_rolls
        else:
            sig = 1
    else:
        sig = err_weighting

    Nptsrc = 0
    Nring = 0
    
    dist_coeffs = {}
    for par in p:
        if par.startswith('ptsrc_dx_'):
            Nptsrc += 1
        elif par.startswith('r0_'):
            Nring += 1
        elif par.startswith('Sci2Idl'):
            dist_coeffs[par] = pdict.pop(par)
            
    if len(dist_coeffs) == 0:
        siaf_aper = wdb.siaf_aper
        ptsrc_ext = 2
    else:
        siaf_aper = get_siaf_aper(wdb._aperturename, wdb._instrument, Sci2Idl_coeffs=dist_coeffs, fit_osamp=1)
        ptsrc_ext = 0
        
    autoscale = []
    modelcubes = []
    modellabels = []
    fmreducs = []
    modelcoeffs0 = []

    if Nptsrc != 0:
        npad = (3 if (ptsrc_ext==0) else 0)
        ptsrc_params = []
        blur_params = {}
        for i in range(Nptsrc):
            ptsrc_params.append(dict(dx=pdict.pop(f'ptsrc_dx_{i+1}'),
                                     dy=pdict.pop(f'ptsrc_dy_{i+1}'),
                                     flux=pdict.pop(f'ptsrc_flux_{i+1}', 1)))
            if (f'ptsrc_detxsig_{i+1}' in pdict):
                blur_params[f'detxsig_{i+1}']=pdict.pop(f'ptsrc_detxsig_{i+1}')
                blur_params[f'detysig_{i+1}']=pdict.pop(f'ptsrc_detysig_{i+1}')
                blur_params[f'dettheta_{i+1}']=pdict.pop(f'ptsrc_dettheta_{i+1}')
            if (f'ptsrc_skyxsig_{i+1}' in pdict):
                blur_params[f'skyxsig_{i+1}']=pdict.pop(f'ptsrc_skyxsig_{i+1}', 0)
                blur_params[f'skyysig_{i+1}']=pdict.pop(f'ptsrc_skyysig_{i+1}', 0)
                blur_params[f'skytheta_{i+1}']=pdict.pop(f'ptsrc_skytheta_{i+1}', 0)

        ptsrc_modelcubes = generate_nptsrc_model_cube(wdb, ptsrc_params, ptsrc_hduls=ptsrc_hduls, return_components=True, 
                                                      distorted_coords=distorted_coords, siaf_aper=siaf_aper, npad=npad, 
                                                      ext=ptsrc_ext, blur_params=blur_params, **ptsrc_gen_kwargs)
        
        for i,modelcube in enumerate(ptsrc_modelcubes):
            if npad != 0:
                modelcube = modelcube[:, npad:-npad, npad:-npad]
            modelcube[np.isnan(wdb.imcube_sci)] = np.nan
            modelcubes.append(modelcube)
            modellabels.append(f'ptsrc{i+1}')
            if (f'ptsrc_flux_{i+1}' in pdict) and ((p[f'ptsrc_flux_{i+1}'].vary) or (p[f'ptsrc_flux_{i+1}'].value != 1)):
                autoscale.append(False)
            else:
                autoscale.append(True)
            modelcoeffs0.append(pdict.pop(f'ptsrc_flux_{i+1}', 1))

    if Nring != 0:
        raw_model_pxscale = wdb.pxscale/disk_model_osamp
        raw_model_center = c_to_c_osamp(wdb.c_star, disk_model_osamp)
        raw_disk_components = disk_model_osamp**2 * grater_nring_2hg_disk_model(**pdict, cent=raw_model_center,
                                                                               distance=distance, nx=wdb.nx*disk_model_osamp, ny=wdb.ny*disk_model_osamp, pxscale=raw_model_pxscale,
                                                                               rmax_accuracy=rmax_accuracy, halfNbSlices=halfNbSlices, return_components=True)

        for i,raw_model in enumerate(raw_disk_components):
            modelcube = wdb.convolver.convolve_model(raw_model, pxscale_in=raw_model_pxscale, c_star_in=raw_model_center, distortion_aper=siaf_aper)
            modelcube[np.isnan(wdb.imcube_sci)] = np.nan
            modelcubes.append(modelcube)
            modellabels.append(f'ring{i+1}')
            if ((f'F_{i+1}' in pdict) and ((p[f'F_{i+1}'].vary) or (p[f'F_{i+1}'].value != 1)) or 
                (f'flux_max_{i+1}' in pdict) and ((p[f'flux_max_{i+1}'].vary) or (p[f'flux_max_{i+1}'].value != 1))):
                autoscale.append(False)
            else:
                autoscale.append(True)
            modelcoeffs0.append(1)

    if add_fm_eps: 
        eps = np.random.normal(scale=1e-32, size=(wdb.ny, wdb.nx))
    else:
        eps = 0

    for modelcube in modelcubes:
        wdb.set_circumstellar_model(modelcube+eps)
        fmrdi = wdb.run_rdi(forward_model=True, save_products=False, collapse_rolls=True, derotate=rdi_reduc.derotated, correct_distortion=rdi_reduc.distortion_corrected)
        fmreducs.append(fmrdi)

    modelcubes = deepcopy([*modelcubes_in, *modelcubes])
    modellabels = deepcopy([*modellabels_in, *modellabels])
    fmreducs = deepcopy([*fmreducs_in, *fmreducs])
    autoscale = np.array([*autoscale_in, *autoscale]).astype(bool)
    modelcoeffs = np.array([*np.repeat(1, len(modelcubes_in)), *modelcoeffs0], dtype=rdi_reduc.rolls.dtype)

    fmimages = np.array([r.rolls for r in fmreducs])

    # fixed model component
    if np.any(~autoscale):
        fmrdi_fixed = np.tensordot(modelcoeffs[~autoscale], fmimages[~autoscale], axes=(0,0))
    else:
        fmrdi_fixed = 0

    if np.any(autoscale):
        if roi.ndim == 3:
            scale_optzone = np.all(roi, axis=0)
        else:
            scale_optzone = roi
        autoscale_coeffs = compute_rdi_coefficients((rdi_reduc.rolls-fmrdi_fixed)[np.newaxis, :]/sig,
                                                    fmimages[autoscale]/sig,
                                                    [scale_optzone])
        
        modelcoeffs[autoscale] = np.median(autoscale_coeffs, axis=(0,1,3))

    fmrdi_composite = np.tensordot(modelcoeffs, fmimages, axes=(0,0))

    if return_soln:
        fmrdi_composite_reduc = deepcopy(fmreducs[0])
        if rdi_reduc.derotated:
            fmrdi_composite_reduc.im *= 0
        for i,fmreduc in enumerate(fmreducs):
            modelcubes[i] *= modelcoeffs[i]
            if fmreduc.rolls is not None:
                fmreduc.rolls *= modelcoeffs[i]
            if rdi_reduc.derotated:
                fmreduc.im *= modelcoeffs[i]
            fmreduc.reduc_label = fmreduc.reduc_label+f', {modellabels[i]}'
            new_output_ext = f'{rdi_reduc.output_ext}_{modellabels[i]}_fwdmod'
            fmreduc.filename = fmreduc.filename.replace(f'{fmreduc.output_ext}'+'_i2d.fits', new_output_ext+'_i2d.fits')
            fmreduc.output_ext = new_output_ext
            fmreducs[i] = fmreduc
            if rdi_reduc.derotated:
                fmrdi_composite_reduc.im += fmreduc.im

        if Nring != 0:
            raw_disk_components = image_manip.frebin(raw_disk_components, scale=1/disk_model_osamp, total=False)
            for i in range(Nring):
                raw_disk_components[i] *= modelcoeffs[np.array(modellabels) == f'ring{i+1}'][0]
        else:
            raw_disk_components = None

        fmrdi_composite_reduc.rolls = fmrdi_composite

        modelcube_composite = np.sum(modelcubes, axis=0)
        wdb.set_circumstellar_model(modelcube_composite)
        return modelcoeffs, modellabels, modelcubes, fmreducs, fmrdi_composite_reduc, modelcube_composite, raw_disk_components, autoscale

    res = ((rdi_reduc.rolls - fmrdi_composite)/sig)[..., roi]

    if q_clip is None:
        res = np.abs(res)
    else:
        low,upp = np.nanpercentile(res, q_clip)
        res = np.abs(res[(res >= low) & (res <= upp)])

    if clear_each_call:
        clear_output()

    if count:
        global counter
        counter += 1
        print('Models evaluated: {0: <16}'.format(counter), end='\r')

    return res


class EdgeOnSafeScatteredLightDisk(ScatteredLightDisk):
    """
    Subclass of VIP's ScatteredLightDisk that is safer+faster for highly inclined disks.

    Changes relative to VIP:
      1. check_inclination() is a dummy method and does not modify self.itilt.
      2. compute_scattered_light() does not exclude the central projected pixels
         with VIP's lmin2 > 0 criterion.
      3. The line-of-sight bounds are computed geometrically, allowing exact or
         near-exact edge-on inclinations without dividing by cos(i).
      4. The pixel(s) containing the star can be sub-sampled to estimate a
         pixel-averaged central flux rather than a point-sampled value.
    """

    def check_inclination(self):
        """Do not alter the inclination so we don't break optimizers."""
        return None

    def _los_nodes(self, halfNbSlices, lwidth=100.0):
        """
        Reproduce VIP's nonuniform line-of-sight sampling variable, but as a
        dimensionless coordinate u in [-1, 1].
        """
        halfNbSlices = int(halfNbSlices)
        if halfNbSlices < 2:
            raise ValueError("halfNbSlices must be >= 2")

        tmp = (
            np.exp(np.arange(halfNbSlices) * np.log(lwidth + 1.0) / (halfNbSlices - 1.0))
            - 1.0
        ) / lwidth

        return np.concatenate((-tmp[:0:-1], tmp))

    def _line_of_sight_interval(
        self,
        x_map,
        y_map,
        vertical_extent_factor=2.0,
        eps=1e-12,
    ):
        """
        Return lower/upper line-of-sight limits for each projected point.

        Coordinates are in the PA-rotated sky frame used internally by VIP:
          x_map : projected disk-frame x coordinate, in au
          y_map : projected disk-frame y coordinate, in au

        Disk coordinates along line of sight l are:
          x_d = x_map
          y_d = cos(i) * y_map + sin(i) * l
          z_d = -sin(i) * y_map + cos(i) * l

        We intersect:
          r_d <= rmax
          |z_d| <= vertical_extent_factor * zmax

        The vertical factor defaults to 2 to mimic the effective VIP integration
        extent, because VIP computes dl_map = lzp - lzm and samples lz0 ± dl_map.
        """
        x_map = np.asarray(x_map, dtype=float)
        y_map = np.asarray(y_map, dtype=float)

        calc = self.dust_density.dust_distribution_calc
        rmax = float(calc.rmax)
        zlim = float(vertical_extent_factor) * float(calc.zmax)

        s = float(self.sini)
        c = float(self.cosi)

        lo = np.full(x_map.shape, -np.inf, dtype=float)
        hi = np.full(x_map.shape, +np.inf, dtype=float)

        # Radial condition:
        # x_d^2 + y_d^2 <= rmax^2
        #
        # x^2 + (c y + s l)^2 <= rmax^2
        # A l^2 + B l + C <= 0
        if abs(s) < eps:
            # Pole-on limit: y_d does not depend on l, so this condition either
            # accepts or rejects the whole sightline.
            radial_valid = (x_map**2 + (c * y_map) ** 2) <= rmax**2
        else:
            A = s**2
            B = 2.0 * c * s * y_map
            C = x_map**2 + (c * y_map) ** 2 - rmax**2

            disc = B**2 - 4.0 * A * C
            radial_valid = disc >= 0.0

            sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
            l1 = (-B - sqrt_disc) / (2.0 * A)
            l2 = (-B + sqrt_disc) / (2.0 * A)

            lo = np.maximum(lo, np.minimum(l1, l2))
            hi = np.minimum(hi, np.maximum(l1, l2))

        # Vertical condition:
        # |z_d| = |-s y + c l| <= zlim
        if abs(c) < eps:
            # Edge-on limit: z_d is independent of l.
            vertical_valid = np.abs(-s * y_map) <= zlim
        else:
            l1 = (s * y_map - zlim) / c
            l2 = (s * y_map + zlim) / c

            lo = np.maximum(lo, np.minimum(l1, l2))
            hi = np.minimum(hi, np.maximum(l1, l2))

            vertical_valid = np.ones(x_map.shape, dtype=bool)

        valid = (
            radial_valid
            & vertical_valid
            & np.isfinite(lo)
            & np.isfinite(hi)
            & (hi > lo)
        )

        return lo, hi, valid

    def _compute_unscaled_at_projected_coords(
        self,
        x_map,
        y_map,
        halfNbSlices=25,
        min_star_distance_au=None,
        vertical_extent_factor=2.0,
        lwidth=100.0,
    ):
        """
        Compute the un-normalized scattered-light image at arbitrary projected
        sky/disk-frame coordinates.

        Excludes the projected lmin2 > 0 cut that removes the star-containing pixel.
        """
        x_map = np.asarray(x_map, dtype=float)
        y_map = np.asarray(y_map, dtype=float)

        if x_map.shape != y_map.shape:
            raise ValueError("x_map and y_map must have matching shapes")

        if min_star_distance_au is None:
            # Finite unresolved-source regularization. This only matters for
            # sightlines that pass extremely close to the star.
            min_star_distance_au = 0.5 * float(self.pxInAU)

        r2_floor = float(min_star_distance_au) ** 2

        lo, hi, valid = self._line_of_sight_interval(
            x_map,
            y_map,
            vertical_extent_factor=vertical_extent_factor,
        )

        out = np.zeros(x_map.shape, dtype=float)
        if not np.any(valid):
            return out

        u = self._los_nodes(halfNbSlices=halfNbSlices, lwidth=lwidth)

        xv = x_map[valid]
        yv = y_map[valid]
        l_mid = 0.5 * (lo[valid] + hi[valid])
        l_half = 0.5 * (hi[valid] - lo[valid])

        s = float(self.sini)
        c = float(self.cosi)
        omega_rad = np.deg2rad(self.omega)

        acc = np.zeros(xv.shape, dtype=float)
        prev = None
        prev_u = None

        for uu in u:
            l = l_mid + uu * l_half

            xd = xv
            yd = c * yv + s * l
            zd = -s * yv + c * l

            d2star = xd**2 + yd**2 + zd**2
            d2star_safe = np.maximum(d2star, r2_floor)
            dstar_safe = np.sqrt(d2star_safe)

            rstar = np.sqrt(xd**2 + yd**2)
            theta_star = np.arctan2(yd, xd)

            cosphi = (
                rstar * s * np.sin(theta_star) + zd * c
            ) / dstar_safe
            cosphi = np.clip(cosphi, -1.0, 1.0)

            r_disk = np.sqrt((xd - self.xdo) ** 2 + (yd - self.ydo) ** 2)
            theta_disk = np.arctan2(yd - self.ydo, xd - self.xdo)
            costheta = np.cos(theta_disk - omega_rad)

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                rho = self.dust_density.density_cylindrical(
                    r_disk,
                    costheta,
                    zd,
                )
                phase = self.phase_function.compute_phase_function_from_cosphi(
                    cosphi
                )
                cur = rho * phase / d2star_safe

            cur = np.nan_to_num(cur, nan=0.0, posinf=0.0, neginf=0.0)

            if prev is not None:
                acc += (uu - prev_u) * (prev + cur)

            prev = cur
            prev_u = uu

        # Trapezoid rule in u, with dl = l_half du.
        out[valid] = 0.5 * acc * l_half * self.pxInAU**2
        return out

    def _star_pixel_mask(self):
        """
        Pixel or pixels whose area contains the stellar position.

        This is done in the unrotated image frame, not the disk-PA frame.
        For odd-sized images this is usually one pixel; for even-sized images
        it may be four pixels depending on VIP's center convention.
        """
        half_px = 0.5 * float(self.pxInAU)
        return (
            (np.abs(self.x_map_0PA) <= half_px)
            & (np.abs(self.y_map_0PA) <= half_px)
        )

    def compute_scattered_light(
        self,
        halfNbSlices=25,
        star_pixel_subsampling=5,
        min_star_distance_au=None,
        vertical_extent_factor=2.0,
        lwidth=100.0,
    ):
        """
        Compute scattered light without forcing the inclination away from edge-on.

        Parameters
        ----------
        halfNbSlices : int
            Same role as in VIP: half-number of line-of-sight samples.

        star_pixel_subsampling : int
            If >1, replace the star-containing pixel(s) with a subpixel-averaged
            estimate. A value of 5 means 25 rays per star-containing pixel.

        min_star_distance_au : float or None
            Small-distance regularization for the 1/r^2 illumination term. If
            None, uses 0.5 pixel in au. This prevents numerical singularities
            when a line of sight passes exactly through the star.

        vertical_extent_factor : float
            Multiple of VIP's zmax used for vertical line-of-sight limits.
            The default 2.0 approximately matches the effective extent used by
            the original VIP sampling.

        lwidth : float
            Controls nonuniform line-of-sight sampling, matching VIP's convention.

        Returns
        -------
        scattered_light_map : 2D ndarray
            Disk scattered-light image.
        """
        # Main point-sampled image.
        image = self._compute_unscaled_at_projected_coords(
            self.x_map,
            self.y_map,
            halfNbSlices=halfNbSlices,
            min_star_distance_au=min_star_distance_au,
            vertical_extent_factor=vertical_extent_factor,
            lwidth=lwidth,
        )

        # Optional pixel-area estimate for the star-containing pixel(s).
        nsub = int(star_pixel_subsampling)
        if nsub > 1:
            mask = self._star_pixel_mask()
            if np.any(mask):
                x0 = self.x_map_0PA[mask]
                y0 = self.y_map_0PA[mask]

                vals = np.zeros(x0.shape, dtype=float)
                offsets = (np.arange(nsub, dtype=float) + 0.5) / nsub - 0.5

                for dy in offsets:
                    for dx in offsets:
                        x_sky = x0 + dx * self.pxInAU
                        y_sky = y0 + dy * self.pxInAU

                        # Apply the same PA rotation as ScatteredLightDisk.set_pa().
                        y_disk_projected = (
                            self.cospa * x_sky + self.sinpa * y_sky
                        )
                        x_disk_projected = (
                            -self.sinpa * x_sky + self.cospa * y_sky
                        )

                        vals += self._compute_unscaled_at_projected_coords(
                            x_disk_projected,
                            y_disk_projected,
                            halfNbSlices=halfNbSlices,
                            min_star_distance_au=min_star_distance_au,
                            vertical_extent_factor=vertical_extent_factor,
                            lwidth=lwidth,
                        )

                image[mask] = vals / float(nsub * nsub)

        # Match VIP's optional max-flux normalization.
        if self.flux_max is not None:
            maxval = np.nanmax(image)
            if np.isfinite(maxval) and maxval > 0:
                image = image * (self.flux_max / maxval)

        self.scattered_light_map = np.asarray(image, dtype=float)
        return self.scattered_light_map


def grater_2hg_disk_model(r0, h0, ain, aout, pa, incl, g1, g2, wg1,
                          e=0., omega=0., gamma=2., beta=1., xdo=0, ydo=0,
                          cent=np.array([160.,160.]), distance=10., nx=320, ny=320,
                          pxscale=0.063*u.arcsec/u.pixel, accuracy=None, rmax_accuracy=None,
                          halfNbSlices=25, polar=False, flux_max=None, return_object=False):
    """
    A simple ring-like disk morphology based on Augereau et al. (1999) and assuming a linear combo of two H-G SPFs as the scattering phase function.

    r0: fiducial radius in au
    h0: technically h0/r0 — the ratio of scale height to fiducial radius at the fiducial radius
    ain: radial density power law exponent interior to r0
    aout: radial density power law exponent exterior to r0
    pa: disk position angle in degrees
    incl: disk inclination in degrees
    g1: 1st Henyey-Greenstein asymmetry parameter
    g2: 2nd Henyey-Greenstein asymmetry parameter
    wg1: Weight for the SPF term with asymmetry parameter g1 (value in range 0-1); wg2 is 1-wg1
    e: eccentricity
    omega: argument of pericenter in degrees
    gamma: vertical density exponent (gamma = 2 for gaussian)
    beta: disk radial flaring exponent (beta = 1 for linear)

    cent: the pixel position for the center of the disk (generally the location of the star in the data)
    distance: distance to the target in parsecs
    nx: number of x-axis pixels for the image
    ny: number of y-axis pixels for the image
    pxscale: the pixel scale for the data; either a float (must be arcsec/pixel) or astropy units (any units that can be cast to arcsec/pixel)
    accuracy: the numerical accuracy for the model; pixels with density below this value will be set to zero
    rmax_accuracy: if accuracy is None, the model's accuracy is set such that non-zero values are achieved to this separation (in au)
    halfNbSlices: the number of planar slices to compute above and below the disk midplane
    polar: if True, a simple bell-shaped polarization curve is used to generate a polarized intensity image
    flux_max: if not None, normalize the model image so that this is the maximum value.
    """

    if accuracy is None:
        if rmax_accuracy is None:
            im_corner_dists = np.hypot(*(np.array([[0, 0], [0, ny-1], [nx-1, 0], [nx-1, ny-1]])-cent).T)
            rmax_accuracy = ang_size_to_proj_sep(px_size_to_ang_size(np.max(im_corner_dists), pxscale), distance).value
        accuracy = (rmax_accuracy/r0)**(aout)

    spf = {'name':'DoubleHG', 'g': [g1, g2], 'weight': wg1, 'polar': polar}
    dens = {'name': '2PowerLaws', 'ain': ain, 'aout': aout, 'a': r0, 'ksi0':h0*r0,
            'e': e, 'gamma': gamma, 'beta': 0, 'accuracy': accuracy} # Setting beta to zero here and then setting below to avoid < 0 beta error message.

    disk = EdgeOnSafeScatteredLightDisk(nx=nx, ny=ny, distance=distance,
                                  itilt=incl, pxInArcsec=(pxscale << u.arcsec/u.pixel).value,
                                  pa=pa-180., omega=omega, xdo=xdo, ydo=ydo,
                                  density_dico=dens, spf_dico=spf, flux_max=flux_max,
                                  xs=cent[0], ys=cent[1])

    disk.dust_density.dust_distribution_calc.rmax = disk.dust_density.dust_distribution_calc.a*accuracy**(1/disk.dust_density.dust_distribution_calc.aout)
    disk.dust_density.dust_distribution_calc.beta = beta
    
    if return_object:
        return disk
    
    image = disk.compute_scattered_light(halfNbSlices=halfNbSlices)
    return image


def grater_nring_2hg_disk_model(cent=np.array([160.,160.]),
                                 distance=10., nx=320, ny=320,
                                 pxscale=0.063*u.arcsec/u.pixel, accuracy=None, rmax_accuracy=None,
                                 polar=False, halfNbSlices=25, return_components=False, **disk_params):

    images = []
    all_rings_finished = False
    i=1
    while not all_rings_finished:
        ring_params = {}
        suffix = f'_{i}'
        for pkey in disk_params:
            if pkey.endswith(suffix):
                ring_params[pkey.replace(suffix, '')] = disk_params[pkey]

        if len(ring_params) == 0:
            all_rings_finished = True

        else:
            F_i = ring_params.pop('F', 1)
            if isinstance(rmax_accuracy, list) or isinstance(rmax_accuracy, np.core.ndarray):
                images.append(F_i * grater_2hg_disk_model(**ring_params, cent=cent,
                                                                distance=distance, nx=nx, ny=ny,
                                                                pxscale=pxscale, accuracy=accuracy, 
                                                                rmax_accuracy=rmax_accuracy[i-1],
                                                                polar=polar, halfNbSlices=halfNbSlices))
            else:
                images.append(F_i * grater_2hg_disk_model(**ring_params, cent=cent,
                                                distance=distance, nx=nx, ny=ny,
                                                pxscale=pxscale, accuracy=accuracy, 
                                                rmax_accuracy=rmax_accuracy,
                                                polar=polar, halfNbSlices=halfNbSlices))
        i+=1
    
    if return_components:
        return np.array(images)
    
    if len(images) == 0:
        return np.zeros((ny,nx))
    
    return np.nansum(images, axis=0)


def make_region_of_interest_mask(reduc, spacerdi, rmin=0., rmax=np.inf, dxy_northup=None, rexcl_northup=None, dxy_detector=None, rexcl_detector=None, units=u.pix):
    """
    Make a region of interest mask.

    Parameters
    ----------
    reduc : object
        winnie.space.SpaceReduction object
    spacerdi : object
        winnie.space.SpaceRDI object
    rmin : float, optional
        Minimum radius for the region of interest.
    rmax : float, optional
        Maximum radius for the region of interest.
    dxy_northup : array_like, optional
        Positions to exclude in the north-up frame relative to the host star's position.
    rexcl_northup : array_like, optional
        Exclusion radii corresponding to `dxy_northup`.
    dxy_detector : array_like, optional
        Positions to exclude in the detector frame (non-derotated) relative to the host star's position.
    rexcl_detector : array_like, optional
        Exclusion radii corresponding to `dxy_detector`.
    units : astropy.units, optional
        Units of the input parameters.

    Returns
    -------
    roi : ndarray
        Boolean mask of the region of interest.
    """
    rmap = dist_to_pt(reduc.c_star, reduc.nx, reduc.ny)
    
    if (dxy_northup is not None) and np.size(rexcl_northup) == 1:
        rexcl_northup = np.array([rexcl_northup]*len(dxy_northup))
    elif dxy_northup is None:
        dxy_northup = rexcl_northup = np.array([])
        
    if (dxy_detector is not None) and np.size(rexcl_detector) == 1:
        rexcl_detector = np.array([rexcl_detector]*len(dxy_detector))
    elif dxy_detector is None:
        dxy_detector = rexcl_detector = np.array([])
        
    if units != u.pix:
        rmin, rmax, dxy_northup, dxy_detector, rexcl_northup, rexcl_detector = [ang_size_to_px_size(x*units, spacerdi.pxscale).value for x in [rmin, rmax, dxy_northup, dxy_detector, rexcl_northup, rexcl_detector]]
        
    roi = (rmap >= rmin) & (rmap <= rmax) & np.all(np.isfinite(reduc.rolls), axis=0)
        
    for i, dxy_i in enumerate(dxy_northup):
        xypos0 = dxy_i + reduc.c_star
        if reduc.derotated:
            roi[dist_to_pt(xypos0, reduc.nx, reduc.ny) <= rexcl_northup[i]] = False
        else:
            for posang in spacerdi._posangs_sci:
                xypos = xy_polar_ang_displacement(*xypos0, -posang, *reduc.c_star)
                roi[dist_to_pt(xypos, reduc.nx, reduc.ny) <= rexcl_northup[i]] = False
                
    for i, dxy_i in enumerate(dxy_detector):
        xypos0 = dxy_i + reduc.c_star
        if not reduc.derotated:
            roi[dist_to_pt(xypos0, reduc.nx, reduc.ny) <= rexcl_detector[i]] = False
        else:
            for posang in spacerdi._posangs_sci:
                xypos = xy_polar_ang_displacement(*xypos0, posang, *reduc.c_star)
                roi[dist_to_pt(xypos, reduc.nx, reduc.ny) <= rexcl_detector[i]] = False
                
    return roi