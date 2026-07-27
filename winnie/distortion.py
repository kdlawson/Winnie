import astropy.units as u
from scipy.interpolate import RegularGridInterpolator
from .utils import xy_polar_ang_displacement, c_to_c_osamp
import numpy as np
from copy import deepcopy
from scipy.interpolate import RegularGridInterpolator
from sklearn.preprocessing import PolynomialFeatures
import pysiaf
import scipy.linalg as linalg
from numpy.polynomial.legendre import leggauss


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


def _make_gauss_legendre_sci_grid(nx, ny, order):
    """
    Construct a tensor-product Gauss-Legendre quadrature grid covering the
    complete science-coordinate domain out to pixel edges in 1-indexed SIAF SCI
    coordinates, i.e.:
        x_sci in [0.5, nx + 0.5] 
        y_sci in [0.5, ny + 0.5]

    Parameters
    ----------
    nx, ny : int
        Science-frame array dimensions.

    order : int
        Number of Gauss-Legendre nodes per axis.

    Returns
    -------
    xsci, ysci : 1D ndarray
        Flattened quadrature-node coordinates in 1-indexed SIAF SCI
        coordinates.

    weights : 1D ndarray
        Flattened tensor-product quadrature weights. The weights sum to nx *
        ny, the area of the fitted science-coordinate domain.
    """
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer.")

    # Standard Gauss-Legendre nodes and weights on [-1, 1].
    tx, wx = leggauss(order)
    ty, wy = leggauss(order)

    # Map x nodes from [-1, 1] to [0.5, nx + 0.5].
    xlo, xhi = 0.5, float(nx) + 0.5
    xsci_1d = (0.5 * (xhi + xlo) + 0.5 * (xhi - xlo) * tx)
    wx = 0.5 * (xhi - xlo) * wx

    # Map y nodes likewise
    ylo, yhi = 0.5, float(ny) + 0.5
    ysci_1d = (0.5 * (yhi + ylo) + 0.5 * (yhi - ylo) * ty)
    wy = 0.5 * (yhi - ylo) * wy

    xsci, ysci = np.meshgrid(xsci_1d, ysci_1d)

    weights_2d = np.multiply.outer(wy, wx)

    return xsci.ravel(), ysci.ravel(), weights_2d.ravel()


def _fit_poly_with_constraints(X, Y, known_coeffs, weights):
    n_features = X.shape[1]
    all_indices = np.arange(n_features)

    fixed_idx = np.array(sorted(known_coeffs), dtype=int)
    free_idx = np.setdiff1d(all_indices, fixed_idx)

    X_fixed = X[:, fixed_idx]
    X_free = X[:, free_idx]
    
    fixed_values = np.array([known_coeffs[i] for i in fixed_idx])
    Y_adjusted = Y - X_fixed @ fixed_values
    
    coeffs = np.zeros(n_features, dtype=np.float64)
    coeffs[fixed_idx] = fixed_values

    sqrt_weights = np.sqrt(weights / weights.sum())
    X_weighted = X_free * sqrt_weights[:, np.newaxis]
    Y_weighted = Y_adjusted * sqrt_weights

    # Solve the weighted system
    coeffs[free_idx], *_ = linalg.lstsq(X_weighted, Y_weighted, check_finite=False, lapack_driver="gelsd")
    
    return coeffs


def enforce_reversible_transforms(aper, use_quad_grid=True, quad_order=None, strict_local_inverse=None, print_values=False):
    """
    Fit Idl2Sci coefficients that minimize residuals for an inverse transform,
    then set them for the PySIAF aperture.

    Parameters
    ----------
    aper : pysiaf aperture
        Aperture whose Sci2Idl coefficients define the forward transform.
        Idl2Sci coefficients are modified in place.
    use_quad_grid : bool
        If True, fit to weighted samples drawn from a Gauss-Legendre quadrature
        grid spanning the science detector positions. If False, use uniform
        weights and fit to the center point of every detector pixel. Use of the
        quadrature grid should be substantially faster at no cost; the detector
        grid sampling method is left primarily as a sanity check. 
    quad_order : int or None
        Number of quadrature nodes per science-coordinate axis. If None, uses
        degree**2 + 1, which exactly integrates the squared closure residual
        when the forward and inverse models both have `degree`. For degree=5,
        this gives quad_order=26.
    strict_local_inverse : bool, optional
        If True, enforce that the fitted Idl2Sci transform is a strict local
        inverse of the Sci2Idl transform at the reference point. This sets bias
        terms to zero (by convention) and fixes the values of first order
        Idl2Sci terms to be the inverse of the Jacobian of the Sci2Idl
        transform. This option may be preferable when the reference point has
        physical significance (e.g., the position of the mask for
        coronagraphy). Defaults to False for FULL apertures and True otherwise.
    print_values : bool
        If True, prints the fit inverse coefficients.

    Returns
    -------
    aper
        The input aperture, modified in place.
    """
    degree = int(aper.Sci2IdlDeg)
    if strict_local_inverse is None:
        strict_local_inverse = 'FULL' not in aper.AperName

    if use_quad_grid:
        if quad_order is None:
            quad_order = degree**2 + 1

        xsci, ysci, weights = _make_gauss_legendre_sci_grid(aper.XSciSize, 
                                                            aper.YSciSize, 
                                                            quad_order)
        
    else:
        ysci, xsci = np.indices((aper.YSciSize, aper.XSciSize), dtype=np.float64)+1
        ysci, xsci = ysci.ravel(), xsci.ravel()
        weights = np.ones_like(xsci)

    # sci -> idl for sample points
    xidl, yidl = aper.sci_to_idl(xsci, ysci)

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    XY_poly = poly.fit_transform(np.column_stack((xidl, yidl)))
    
    # Set constraints for coefficients we can determine analytically / by convention: 
    if strict_local_inverse:
        jac = np.array([[aper.Sci2IdlX10, aper.Sci2IdlX11],
                        [aper.Sci2IdlY10, aper.Sci2IdlY11]])

        jac_inv = np.linalg.inv(jac)

        known_coeffs_x = {0: 0.0, 
                          1: jac_inv[0,0],
                          2: jac_inv[0,1]}
        
        known_coeffs_y = {0: 0.0, 
                          1: jac_inv[1,0],
                          2: jac_inv[1,1]}

    else:
        known_coeffs_x = {0: 0.0}
        known_coeffs_y = {0: 0.0}

    coeffs_inv_x = _fit_poly_with_constraints(XY_poly, xsci - aper.XSciRef, known_coeffs_x, weights)
    coeffs_inv_y = _fit_poly_with_constraints(XY_poly, ysci - aper.YSciRef, known_coeffs_y, weights)

    xcoeff_names = [f"Idl2SciX{i}{j}" for i in range(degree+1) for j in range(i+1)]
    ycoeff_names = [f"Idl2SciY{i}{j}" for i in range(degree+1) for j in range(i+1)]

    for name, value in zip(xcoeff_names, coeffs_inv_x):
        setattr(aper, name, float(value))
        if print_values:
            print(name, value)

    for name, value in zip(ycoeff_names, coeffs_inv_y):
        setattr(aper, name, float(value))
        if print_values:
            print(name, value)

    return aper


def get_siaf_aper(aperturename, instrument='NIRCam', Sci2Idl_coeffs={}, enforce_reversability=True, use_quad_grid=True, quad_order=None, strict_local_inverse=None, print_values=False):
    """
    Fetches the pySIAF aperture for aperturename and instrument, then updates
    any coefficients specified in Sci2Idl_coeffs. Then, if
    enforce_reversability is True, fits the corresponding Idl2Sci coefficients
    and sets them to ensure that coordinate transforms are reversible.

    Parameters
    ----------
    aperturename : str
        Name of the SIAF aperture.
    instrument : str, optional
        Name of the instrument (default is 'NIRCam').
    Sci2Idl_coeffs : dict, optional
        Dictionary of Sci2Idl coefficients to update in the SIAF aperture (for
        departing from the nominal distortion model).
    enforce_reversability : bool, optional
        If True, enforces that the sci <-> idl coordinate transforms are
        reversible (default is True).
    use_quad_grid : bool, optional
        If True, uses a Gauss-Legendre quadrature grid for fitting inverse
        (default is True).
    use_quad_grid : bool
        If True, fit to weighted samples drawn from a Gauss-Legendre quadrature
        grid spanning the science detector positions. If False, use uniform
        weights and fit to the center point of every detector pixel. Use of the
        quadrature grid should be substantially faster at no cost; the detector
        grid sampling method is left primarily as a sanity check. 
    quad_order : int or None
        Number of quadrature nodes per science-coordinate axis. If None, uses
        degree**2 + 1, which exactly integrates the squared closure residual
        when the forward and inverse models both have `degree`. For degree=5,
        this gives quad_order=26.
    strict_local_inverse : bool, optional
        If True, enforce that the fitted Idl2Sci transform is a strict local
        inverse of the Sci2Idl transform at the reference point. This sets bias
        terms to zero (by convention) and fixes the values of first order
        Idl2Sci terms to be the inverse of the Jacobian of the Sci2Idl
        transform. This option may be preferable when the reference point has
        physical significance (e.g., the position of the mask for
        coronagraphy). Defaults to False for FULL apertures and True otherwise.
    print_values : bool
        If True, prints the fit inverse coefficients.
    """
    aper = deepcopy(pysiaf.Siaf(instrument)[aperturename])
    aper.__dict__.update(Sci2Idl_coeffs)
    if enforce_reversability:
        aper = enforce_reversible_transforms(aper=aper,
                                             use_quad_grid=use_quad_grid, 
                                             quad_order=quad_order, 
                                             strict_local_inverse=strict_local_inverse,
                                             print_values=print_values)
    return aper