import astropy.units as u
from scipy.interpolate import RegularGridInterpolator
from .utils import xy_polar_ang_displacement, c_to_c_osamp
import numpy as np
from copy import deepcopy
from scipy.interpolate import RegularGridInterpolator
from sklearn.preprocessing import PolynomialFeatures
import pysiaf
import scipy.linalg as linalg


def _parse_pxscale(pxscale, siaf_ap):
    if pxscale is None:
        scale = 0.5 * (siaf_ap.XSciScale + siaf_ap.YSciScale)
        xscale = yscale = scale
    elif np.ndim(pxscale) == 0:
        xscale = yscale = pxscale
    else:
        xscale, yscale = pxscale

    xscale = (xscale << u.arcsec/u.pixel).value
    yscale = (yscale << u.arcsec/u.pixel).value

    return xscale, yscale


def _make_coord_grid(shape, osamp=1):
    y, x = np.indices(shape, dtype=float)

    if osamp != 1:
        x, y = c_to_c_osamp(np.array([x, y]), 1 / osamp)

    return x, y


def _undistorted_ref_from_detector_ref(
    siaf_ap,
    c_ref_det,
    c_star,
    xscale,
    yscale,
):
    """
    Given c_ref_det, the detector-frame reference-pixel location,
    compute the corresponding reference location in the undistorted detector
    frame such that c_star remains fixed.

    If c_star is None, this is just c_ref_det.
    """
    c_ref_det = np.asarray(c_ref_det, dtype=float)

    if c_star is None:
        return c_ref_det.copy()

    c_star = np.asarray(c_star, dtype=float)

    c_ref0 = np.array([siaf_ap.XSciRef, siaf_ap.YSciRef], dtype=float) - 1

    x_idl_star, y_idl_star = siaf_ap.sci_to_idl(
        c_star[0] - c_ref_det[0] + c_ref0[0] + 1,
        c_star[1] - c_ref_det[1] + c_ref0[1] + 1,
    )

    c_ref_undist_det = np.array([
        c_star[0] - x_idl_star / xscale,
        c_star[1] - y_idl_star / yscale,
    ])

    return c_ref_undist_det


def _interp_image(image, x_grid, y_grid, x_samp, y_samp,
                  method='cubic', fill_value=0., prop_nans=True, prop_threshold=0.1):
    
    if (prop_nans) and (method not in ['nearest', 'linear']):
        nans = np.isnan(image)
        any_nans = np.any(nans)
        
    if (not prop_nans) or (method in ['nearest', 'linear']) or (not any_nans):
        interpolator = RegularGridInterpolator(
            (y_grid[:, 0], x_grid[0]),
            image,
            method=method,
            bounds_error=False,
            fill_value=fill_value)

        out = interpolator(np.array([y_samp.ravel(), x_samp.ravel()]).T)
        
    else:
        image_fill = np.where(nans, np.nanmedian(image), image)
        
        interpolator = RegularGridInterpolator((y_grid[:, 0], x_grid[0]), image_fill, method=method, bounds_error=False, fill_value=fill_value)
        im_out = interpolator(np.array([y_samp.ravel(), x_samp.ravel()]).T)
        
        interpolator = RegularGridInterpolator((y_grid[:, 0], x_grid[0]), nans.astype(float), method='linear', bounds_error=False, fill_value=0)
        nans_out = interpolator(np.array([y_samp.ravel(), x_samp.ravel()]).T) > prop_threshold
        out = np.where(nans_out, np.nan, im_out)
        
    return out.reshape(image.shape)


def distort_image(
    image,
    siaf_ap,
    c_ref,
    pxscale_in=None,
    posang_out=0,
    osamp=1,
    method='cubic',
    c_star=None,
    fill_value=0.,
    return_info=False
):
    """
    Apply SIAF distortion to a north-up undistorted input image and orient it
    to posang_out in the detector frame.

    c_ref is always interpreted as the detector-frame reference
    location of the SIAF reference point.

    If c_star is provided, the star remains fixed and the rotation is performed
    about c_star.

    Note: in pratice, we don't typically use posang_out != 0, but it is
    included for completeness and symmetry with undistort_image. Using this in
    forward modeling would require that we convolve with rotated PSFs to save
    the extra interpolation step. Given the typical size of PSF grids,
    rotating then convolving and distorting is likely preferable.

    Returns
    -------
    image_dist : 2D ndarray
        Distorted detector-frame image.

    c_ref_out : ndarray
        Location of the input reference point after rotation + distortion.

    info : dict, optional
        Returned if return_info=True.
    """
    image = np.asarray(image)
    c_ref_det = np.asarray(c_ref, dtype=float)

    xscale_in, yscale_in = _parse_pxscale(pxscale_in, siaf_ap)

    c_ref0 = np.array([siaf_ap.XSciRef, siaf_ap.YSciRef], dtype=float) - 1

    x_grid, y_grid = _make_coord_grid(image.shape, osamp=osamp)

    c_ref_undist_det = _undistorted_ref_from_detector_ref(
        siaf_ap,
        c_ref_det,
        c_star,
        xscale_in,
        yscale_in,
    )

    if c_star is None:
        c_rot = c_ref_undist_det
    else:
        c_rot = np.asarray(c_star, dtype=float)

    # Distorted detector output coordinates -> ideal coordinates.
    x_idl, y_idl = siaf_ap.sci_to_idl(
        x_grid - c_ref_det[0] + c_ref0[0] + 1,
        y_grid - c_ref_det[1] + c_ref0[1] + 1,
    )

    # Ideal coordinates -> intermediate undistorted detector-frame coordinates.
    x_undist_det = x_idl / xscale_in + c_ref_undist_det[0]
    y_undist_det = y_idl / yscale_in + c_ref_undist_det[1]

    # Intermediate detector-frame coordinates -> north-up input coordinates.
    #
    # This is intentionally after the SIAF inverse, not before it.
    if posang_out != 0:
        x_in, y_in = xy_polar_ang_displacement(
            x_undist_det,
            y_undist_det,
            posang_out,
            *c_rot
        )
    else:
        x_in = x_undist_det
        y_in = y_undist_det

    image_dist = _interp_image(
        image,
        x_grid,
        y_grid,
        x_in,
        y_in,
        method=method,
        fill_value=fill_value,
        prop_nans=False
    )

    if posang_out != 0:
        x_ref_undet, y_ref_undet = xy_polar_ang_displacement(
            c_ref_undist_det[0],
            c_ref_undist_det[1],
            -posang_out,
            *c_rot
        )
    else:
        x_ref_undet = c_ref_undist_det[0]
        y_ref_undet = c_ref_undist_det[1]

    x_ref_idl = (x_ref_undet - c_ref_undist_det[0]) * xscale_in
    y_ref_idl = (y_ref_undet - c_ref_undist_det[1]) * yscale_in

    x_ref_sci, y_ref_sci = siaf_ap.idl_to_sci(x_ref_idl, y_ref_idl)

    c_ref_out = np.array([
        x_ref_sci - c_ref0[0] - 1 + c_ref_det[0],
        y_ref_sci - c_ref0[1] - 1 + c_ref_det[1],
    ])

    if return_info:
        info = {
            "c_ref_det": c_ref_det,
            "c_ref_undist_det": c_ref_undist_det,
            "c_ref_out": c_ref_out,
            "c_rot": c_rot,
            "posang_out": posang_out,
        }
        return image_dist, c_ref_out, info

    return image_dist, c_ref_out


def undistort_image(
    image,
    siaf_ap,
    c_ref,
    pxscale_out=None,
    posang_in=0,
    osamp=1,
    method='cubic',
    c_star=None,
    fill_value=0.,
    return_info=False,
):
    """
    Correct astrometric distortion and derotate an input image to north-up in one
    interpolation.

    The input image is assumed to be in the distorted detector frame and has
    not yet been derotated.

    c_ref is always interpreted as the detector-frame/pre-rotation reference
    location of the SIAF reference point.

    The result should broadly match undistort_image followed by derotation to
    north-up as separate operations.

    Returns
    -------
    image_undist : 2D ndarray
        Distortion-corrected and derotated image.

    c_ref_out : ndarray
        Location of the reference point in the final derotated output frame.

    info : dict, optional
        Returned if return_info=True.
    """
    image = np.asarray(image)
    c_ref_det = np.asarray(c_ref, dtype=float)

    xscale_out, yscale_out = _parse_pxscale(pxscale_out, siaf_ap)

    c_ref0 = np.array([siaf_ap.XSciRef, siaf_ap.YSciRef], dtype=float) - 1

    x_grid, y_grid = _make_coord_grid(image.shape, osamp=osamp)

    c_ref_undist_det = _undistorted_ref_from_detector_ref(
        siaf_ap,
        c_ref_det,
        c_star,
        xscale_out,
        yscale_out,
    )

    if c_star is None:
        c_rot = c_ref_undist_det
    else:
        c_rot = np.asarray(c_star, dtype=float)

    if posang_in != 0:
        x_undist_det, y_undist_det = xy_polar_ang_displacement(
            x_grid,
            y_grid,
            -posang_in,
            *c_rot
        )
    else:
        x_undist_det = x_grid
        y_undist_det = y_grid

    # Intermediate undistorted detector-frame pixels -> ideal coordinates.
    x_idl = (x_undist_det - c_ref_undist_det[0]) * xscale_out
    y_idl = (y_undist_det - c_ref_undist_det[1]) * yscale_out

    # Ideal coordinates -> distorted SIAF science coordinates.
    x_sci, y_sci = siaf_ap.idl_to_sci(x_idl, y_idl)

    # SIAF science coordinates -> input distorted image pixel coordinates.
    x_in = x_sci - c_ref0[0] - 1 + c_ref_det[0]
    y_in = y_sci - c_ref0[1] - 1 + c_ref_det[1]

    image_undist = _interp_image(
        image,
        x_grid,
        y_grid,
        x_in,
        y_in,
        method=method,
        fill_value=fill_value,
        prop_nans=True
    )

    # Location of the undistorted detector-frame reference point after the
    # final derotation to north-up.
    if posang_in != 0:
        c_ref_out = np.array(
            xy_polar_ang_displacement(
                c_ref_undist_det[0],
                c_ref_undist_det[1],
                posang_in,
                *c_rot
            )
        )
    else:
        c_ref_out = c_ref_undist_det.copy()

    if return_info:
        info = {
            "c_ref_det": c_ref_det,
            "c_ref_undist_det": c_ref_undist_det,
            "c_ref_out": c_ref_out,
            "c_rot": c_rot,
            "posang_in": posang_in,
        }
        return image_undist, c_ref_out, info

    return image_undist, c_ref_out


def fit_poly_with_constraints(X_poly, target, known_coeffs):
    n_features = X_poly.shape[1]
    all_indices = np.arange(n_features)

    # Mask for unknowns
    fixed_idx = np.array(list(known_coeffs.keys()), dtype=int)
    free_idx = np.setdiff1d(all_indices, fixed_idx)

    # Subtract known terms from target
    X_fixed = X_poly[:, fixed_idx]
    y_adjusted = target - X_fixed @ np.array([known_coeffs[i] for i in fixed_idx])

    # Fit only the free coefficients
    X_free = X_poly[:, free_idx]

    coeffs_full = np.zeros(n_features)
    lu, piv = linalg.lu_factor(X_free.T @ X_free, check_finite=False)
    coeffs_free = linalg.lu_solve((lu, piv), (X_free.T @ (y_adjusted[:, np.newaxis]))[:,0], check_finite=False)
    
    coeffs_full[free_idx] = coeffs_free
    for i, val in known_coeffs.items():
        coeffs_full[i] = val
    return coeffs_full


def enforce_reversible_transforms(aper, fit_osamp=2, xy_sci_validate=None, print_values=False):
    """
    Fits the corresponding Idl2Sci coefficients and sets them to ensure that
    coordinate transforms are reversible (Idl2Sci <-> Sci2Idl) across the
    desired subarray. 
    
    This is intended to fix the issue of discontinuities in the astrometric
    distortion across the NIRCam COM. Likely not ideal for NIRCam coronagraphy
    using atypical subarrays (e.g., FULL). In those cases, distortion likely
    needs to be applied piecewise with a custom procedure.  
    """
    degree = aper.Sci2IdlDeg
    
    nx, ny = aper.XSciSize, aper.YSciSize
    nxfit, nyfit = int(nx*fit_osamp), int(ny*fit_osamp)
    ysci, xsci = c_to_c_osamp(np.indices((nyfit, nxfit), dtype=np.float64), 1/fit_osamp)+1
    ysci, xsci = ysci.flatten(), xsci.flatten()
    xidl, yidl = aper.sci_to_idl(xsci, ysci)

    poly = PolynomialFeatures(degree=degree)
    XY_poly = poly.fit_transform(np.vstack([xidl, yidl]).T)
    
    known_coeffs_x = {0: 0.0, 1: 1./aper.Sci2IdlX10}
    known_coeffs_y = {0: 0.0, 2: 1./aper.Sci2IdlY11}

    xcoeff_names = []
    ycoeff_names = []

    for par in aper.__dict__:
        if par.startswith('Idl2SciX') and (aper.__dict__[par] is not None):
            xcoeff_names.append(par)
        elif par.startswith('Idl2SciY') and (aper.__dict__[par] is not None):
            ycoeff_names.append(par)

    coeffs_inv_x = fit_poly_with_constraints(XY_poly, xsci-aper.XSciRef, known_coeffs_x)
    coeffs_inv_y = fit_poly_with_constraints(XY_poly, ysci-aper.YSciRef, known_coeffs_y)

    for i,s in enumerate(xcoeff_names):
        aper.__dict__[s] = coeffs_inv_x[i]
        if print_values:
            print(s, coeffs_inv_x[i])

    for i,s in enumerate(ycoeff_names):
        aper.__dict__[s] = coeffs_inv_y[i]
        if print_values:
            print(s, coeffs_inv_y[i])
    
    if xy_sci_validate is not None:
        xy_sci_in = np.array(xy_sci_validate)
        xy_idl = aper.sci_to_idl(*xy_sci_in)
        xy_sci_out = aper.idl_to_sci(*xy_idl)
        xy_res = xy_sci_in - xy_sci_out
        
        print(np.nanmedian(xy_res[0]), np.nanmedian(xy_res[1]))
        print(np.nanmedian(np.hypot(*xy_res)))
    return aper
        
    
def get_siaf_aper(aperturename, instrument='NIRCam', Sci2Idl_coeffs={}, enforce_reversability=True, fit_osamp=4, xy_sci_validate=None, print_values=False):
    """
    Fetches the pySIAF aperture for aperturename and instrument, then updates
    any coefficients specified in Sci2Idl_coeffs. Then, if
    enforce_reversability is True, fits the corresponding Idl2Sci coefficients
    and sets them to ensure that coordinate transforms are reversible.
    """
    aper = deepcopy(pysiaf.Siaf(instrument)[aperturename])
    aper.__dict__.update(Sci2Idl_coeffs)
    if enforce_reversability:
        aper = enforce_reversible_transforms(aper, fit_osamp, xy_sci_validate, print_values)
    return aper