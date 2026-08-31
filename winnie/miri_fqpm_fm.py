from tqdm.auto import tqdm
from astropy.io import fits
from .utils import (pad_or_crop_image, xy_polar_ang_displacement, c_to_c_osamp)
from .convolution import apply_partial_dist_to_stpsf_hdul
import webbpsf_ext
import os
from copy import deepcopy
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
import numpy as np
from importlib.resources import files
from .distortion import undistort_image

_RESOURCE_DIR = files("winnie") / 'resources'

def _dtheta_for_r_and_projsep(r,rho):
    "Returns the angle, dtheta, in degrees that will produce azimuthal displacement rho at radial separation r (r and rho in matching units)."
    return 2*np.rad2deg(np.arcsin(rho/(2*r)))


def _logspace_min_step(v1, v2, N, dv, *, endpoint=True, dtype=float):
    """
    Generate approximately logarithmically spaced values between v1 and v2,
    while enforcing that adjacent values differ by at least dv.

    Parameters
    ----------
    v1, v2 : float
        Start and end values. Must be positive for logarithmic spacing.
    N : int
        Number of values to return.
    dv : float
        Minimum allowed absolute spacing between adjacent values.
    endpoint : bool, default=True
        If True, include v2 exactly as the final value.
    dtype : dtype, default=float
        Output dtype.

    Returns
    -------
    vals : ndarray
        Array of length N.

    Raises
    ------
    ValueError
        If the requested spacing is impossible.
    """
    if N < 1:
        raise ValueError("N must be >= 1.")
    if v1 <= 0 or v2 <= 0:
        raise ValueError("v1 and v2 must be positive for logarithmic spacing.")
    if dv < 0:
        raise ValueError("dv must be non-negative.")

    if N == 1:
        return np.array([v1], dtype=dtype)

    reverse = v2 < v1
    lo, hi = (v2, v1) if reverse else (v1, v2)

    if endpoint:
        if (N - 1) * dv > (hi - lo):
            raise ValueError(
                f"Impossible: {N} points with minimum spacing {dv} "
                f"cannot fit between {v1} and {v2}."
            )

        # Ideal logarithmic spacing
        vals = np.geomspace(lo, hi, N)

        # Feasible lower/upper bounds for each index, ensuring room for neighbors
        lower = lo + dv * np.arange(N)
        upper = hi - dv * (N - 1 - np.arange(N))

        # Clip the ideal log-spaced points into the feasible region
        vals = np.clip(vals, lower, upper)

        # Enforce endpoints exactly
        vals[0] = lo
        vals[-1] = hi

    else:
        if N * dv > (hi - lo):
            raise ValueError(
                f"Impossible: {N} points with minimum spacing {dv} "
                f"cannot fit in [{v1}, {v2}) without endpoint."
            )

        # Start with N+1 points and drop the endpoint
        vals = np.geomspace(lo, hi, N + 1)[:-1]

        lower = lo + dv * np.arange(N)
        upper = hi - dv * (N - np.arange(N))

        vals = np.clip(vals, lower, upper)
        vals[0] = lo

    if reverse:
        vals = vals[::-1]

    return vals.astype(dtype)


def nearest_sample_grid(xg, yg, xsamp, ysamp):
    """
    For each (xg, yg) grid point, find the nearest point among
    (xsamp, ysamp).

    Returns
    -------
    xgNearest, ygNearest : 2D arrays
        Same shape as xg/yg, containing the nearest sample coordinates.
    idx : 2D int array
        Index into xsamp/ysamp of the nearest sample.
    dist : 2D array
        Euclidean distance to the nearest sample.
    """
    xg = np.asarray(xg)
    yg = np.asarray(yg)
    xsamp = np.asarray(xsamp)
    ysamp = np.asarray(ysamp)

    sample_points = np.column_stack([xsamp, ysamp])
    grid_points = np.column_stack([xg.ravel(), yg.ravel()])

    tree = cKDTree(sample_points)
    dist_flat, idx_flat = tree.query(grid_points, k=1)

    idx = idx_flat.reshape(xg.shape)
    dist = dist_flat.reshape(xg.shape)

    xgNearest = xsamp[idx]
    ygNearest = ysamp[idx]

    return xgNearest, ygNearest, idx, dist


def get_fqpm_psf_grid_inds(
    c_coron,
    psf_offsets_polar,
    osamp=2,
    shape=None,
    pxscale=None,
    inst=None,
    posang=0,
    c_star=None):
    """
    The goal is to match each pixel in the (oversampled) array with the most similar sampled PSF in terms of
    radial separation and separation from the nearest FQPM boundary. 
    This ends up being slightly different than simply matching to the nearest sampled angular position 
    in a way that's important for FQPM performance. 
    """
    if posang != 0:
        raise NotImplementedError("FQPM PSF index calculations do not yet support creation of derotated index maps.")
    
    siaf_ap = inst.siaf[inst.aperturename]

    if shape is None:
        nx = siaf_ap.XSciSize
        ny = siaf_ap.YSciSize
    else:
        ny, nx = shape

    field_rot = 0 if inst._rotation is None else inst._rotation

    psf_inds = np.full((ny * osamp, nx * osamp), -1, dtype=np.int32)

    c_coron0 = np.array([siaf_ap.XSciRef, siaf_ap.YSciRef]) - 1

    rsamp, thsamp = psf_offsets_polar.copy()
    xsamp, ysamp = np.array(webbpsf_ext.coords.rtheta_to_xy(*psf_offsets_polar))

    yg, xg = c_to_c_osamp(np.indices((ny * osamp, nx * osamp), dtype=rsamp.dtype), 1 / osamp)
    
    xgIdl, ygIdl = siaf_ap.sci_to_idl(xg - c_coron[0] + c_coron0[0] + 1, 
                                      yg - c_coron[1] + c_coron0[1] + 1)

    # A PSF sample is generated for the center of every oversampled pixel within some specified radius r0
    # of the coronagraph center.
    # Step 1: Find any cases where a PSF sample is arbitrarily close to a pixel center >>> pair them;
    # these are the pixels inside r0.
    xgNearest, ygNearest, idxNearest, distNearest = nearest_sample_grid(xgIdl, ygIdl, xsamp, ysamp)

    match = np.isclose(distNearest, 0, atol=1e-3)
    psf_inds[match] = idxNearest[match]

    # For remaining points:
    rg, tg = webbpsf_ext.coords.xy_to_rtheta(xgIdl, ygIdl)
    tg = np.mod(tg, 360)

    rsamp_uni, rcount = np.unique(rsamp, return_counts=True)
    rsamp_uni = rsamp_uni[rcount >= 8]  # Every r 'grid' pos beyond r0 must have at least 8 samples

    remaining = ~match

    th_fqpm = np.array([0.0, 90.0, 180.0, 270.0]) - field_rot
    th_fqpm_wrap = np.array([*th_fqpm - 360.0, *th_fqpm, *th_fqpm + 360.0])
    th_fqpm_wrap0 = np.array([*th_fqpm, *th_fqpm, *th_fqpm])

    nearest_th_fqpm_map = th_fqpm_wrap0[np.argmin(np.abs(tg[None, :, :] - th_fqpm_wrap[:, None, None]), axis=0)]

    if np.any(remaining):
        # Flatten only once. Work with flat indices, then assign back into psf_inds.ravel().
        rem_flat = np.flatnonzero(remaining.ravel())

        rg_flat = rg.ravel()
        tg_flat = tg.ravel()
        nearest_th_flat = nearest_th_fqpm_map.ravel()
        psf_inds_flat = psf_inds.ravel()

        r_rem = rg_flat[rem_flat]

        insert = np.searchsorted(rsamp_uni, r_rem, side="left")

        left = np.clip(insert - 1, 0, len(rsamp_uni) - 1)
        right = np.clip(insert, 0, len(rsamp_uni) - 1)

        dl = np.abs(r_rem - rsamp_uni[left])
        dr = np.abs(rsamp_uni[right] - r_rem)

        nearest_r_ind_rem = np.where(dr < dl, right, left)
        nearest_rval_rem = rsamp_uni[nearest_r_ind_rem]

        # Process by nearest sampled radius, then by quadrant boundary.
        # This avoids constructing a huge Npix x Nsamp array.
        for nearest_rval in np.unique(nearest_rval_rem):
            r_group_sel = nearest_rval_rem == nearest_rval
            r_group_flat = rem_flat[r_group_sel]

            if r_group_flat.size == 0:
                continue

            nearest_r_mask = rsamp == nearest_rval
            sample_inds_for_rval = np.flatnonzero(nearest_r_mask)
            thvals_for_rval = thsamp[nearest_r_mask]

            # Iterate over FQPM bounds.
            for nearest_th_fqpm in th_fqpm:
                gsel = nearest_th_flat[r_group_flat] == nearest_th_fqpm
                gflat = r_group_flat[gsel]

                if gflat.size == 0:
                    continue

                dthvals = np.fmod((thvals_for_rval - nearest_th_fqpm).round(5), 360)
                dthvals = np.where(dthvals >= 180, dthvals - 360.0, dthvals)
                good_th_inds = np.abs(dthvals) <= 45.0

                dthvals_good = dthvals[good_th_inds]
                thvals_good = thvals_for_rval[good_th_inds]
                sample_inds_good = sample_inds_for_rval[good_th_inds]

                r_th_match = np.empty_like(sample_inds_good)
                for k, thval in enumerate(thvals_good):
                    r_th_match[k] = np.where(nearest_r_mask & (thsamp == thval))[0][0]

                dthg_flat = np.fmod(tg_flat[gflat] - nearest_th_fqpm, 360)
                dthg_flat = np.where(dthg_flat > 45., dthg_flat - 360., dthg_flat)

                rhovals = 2. * np.pi * nearest_rval * (dthvals_good / 360.)
                rhog = 2. * np.pi * rg_flat[gflat] * (dthg_flat / 360.)

                rho_ind = np.argmin(np.abs(rhovals[None, :] - rhog[:, None]), axis=1)
                psf_inds_flat[gflat] = r_th_match[rho_ind]
    return psf_inds


def generate_fqpm_psf_grid(inst, source_spectrum=None, shift=None, osamp=2, fov_pixels=201, show_progress=True,
                           c_coron_rolls=None, r0=0.0, nr=10, rmax=5, nrho=10, log_rscale=False, normalize='exit_pupil',
                           rho0=0.1, rho0_rmax=None, log_rhoscale=True, nlambda=None):
    try:
        import stpsf
    except ModuleNotFoundError:
        import webbpsf as stpsf

    psf_offsets, psf_offsets_polar = _generate_fqpm_psf_positions(inst, 
                                                                 osamp=osamp, 
                                                                 c_coron_rolls=c_coron_rolls, 
                                                                 r0=r0, 
                                                                 nr=nr, 
                                                                 rmax=rmax, 
                                                                 log_rscale=log_rscale, 
                                                                 nrho=nrho, 
                                                                 rho0=rho0,
                                                                 rho0_rmax=rho0_rmax,
                                                                 log_rhoscale=log_rhoscale)

    siaf_ap = inst.siaf[inst.aperturename]

    c_coron0 = np.array([siaf_ap.XSciRef, siaf_ap.YSciRef])-1

    iterator = enumerate(tqdm(psf_offsets.T, leave=False)) if show_progress else enumerate(psf_offsets.T)

    inst_grid = deepcopy(inst)

    f_tmp = f'tmp_winnie_fqpm_psfs_{inst.filter}.fits'
    f_tmp_nodist = f'tmp_winnie_fqpm_psfs_{inst.filter}_nodist.fits'

    psfs_shape = (psf_offsets.shape[1], fov_pixels*osamp, fov_pixels*osamp)

    if os.path.exists(f_tmp):
        psfs = fits.getdata(f_tmp)
        if psfs.shape != psfs_shape:
            os.remove(f_tmp)
            psfs = np.zeros(psfs_shape, np.float32)
    else:
        psfs = np.zeros(psfs_shape, np.float32)

    if os.path.exists(f_tmp_nodist):
        psfs_nodist = fits.getdata(f_tmp_nodist)
        if not np.all(psfs_nodist.shape == psfs_shape): 
            os.remove(f_tmp_nodist)
            psfs_nodist = np.zeros(psfs_shape, np.float32)
    else:
        psfs_nodist = np.zeros(psfs_shape, np.float32)

    extra_shift = np.array([psfs_shape[1]%2-1, psfs_shape[2]%2-1])[::-1]/2.
    if shift is None:
        shift = 0

    for i, psf_offset in iterator:
        if not np.all(np.isclose(psfs[i], 0)):
            continue

        psf_offset_px = siaf_ap.idl_to_sci(*psf_offset) - (c_coron0 + 1) # +1 because idl_to_sci is 1-indexed and c_coron0 is 0-indexed as defined above

        # This is to accomodate a nuance in the way STPSF simulates FQPM PSFs.
        # TODO: webbpsf-ext has a much less hamfisted version of this that
        # would save time here.
        fov_pixels0 = int(np.ceil(np.max(fov_pixels+2*np.abs(psf_offset_px)+1)))
        c0 = np.repeat((fov_pixels0-1.)/2., 2)

        inst_grid.options['source_offset_x'] = psf_offset[0]
        inst_grid.options['source_offset_y'] = psf_offset[1]

        inst_grid.detector_position = siaf_ap.idl_to_sci(*psf_offset) # Set det position to get correct field dep WF aberration

        psf_hdul = inst_grid.calc_psf(source=source_spectrum, fov_pixels=fov_pixels0, oversample=osamp, normalize=normalize, nlambda=nlambda)
        psfs[i] = pad_or_crop_image(psf_hdul[2].data, [osamp*fov_pixels, osamp*fov_pixels], order=5, cent=c_to_c_osamp(c0 + psf_offset_px - shift, osamp) - extra_shift)

        psf_hdul = apply_partial_dist_to_stpsf_hdul(psf_hdul, inst_grid)
        psfs_nodist[i] = pad_or_crop_image(psf_hdul[0].data, [osamp*fov_pixels, osamp*fov_pixels], order=5, cent=c_to_c_osamp(c0 + psf_offset_px - shift, osamp) - extra_shift)

        if i%100 == 0: # Save to disk every 100 PSFs in case of interruption
            fits.writeto(f_tmp, psfs, overwrite=True)
            fits.writeto(f_tmp_nodist, psfs_nodist, overwrite=True)

    os.remove(f_tmp)
    os.remove(f_tmp_nodist)

    if np.any(extra_shift != 0): # Crop any even axes to odd
        psfs = psfs[:, :int(extra_shift[1]*2), :int(extra_shift[0]*2)]
        psfs_nodist = psfs_nodist[:, :int(extra_shift[1]*2), :int(extra_shift[0]*2)]

    return psfs, psfs_nodist, psf_offsets_polar, psf_offsets


def _generate_fqpm_psf_positions(inst, osamp=2, c_coron_rolls=None, r0=0.0, rho0=0.0, nr=10, rmax=5, rho0_rmax=None, log_rscale=False, log_rhoscale=False, nrho=10):

    siaf_ap = inst.siaf[inst.aperturename]
    nx, ny = siaf_ap.XSciSize, siaf_ap.YSciSize

    drmin = inst.pixelscale/osamp

    # Set up the PSF grid.
    if r0 >= rmax:
        rvals = np.array([r0])
    else:
        if log_rscale:
            rvals = _logspace_min_step(drmin, rmax-r0, nr, drmin, dtype=np.float32)+r0
        else:
            rvals = np.linspace(drmin, rmax-r0, nr)+r0

    if rho0_rmax is None:
        rho0_rmax = rmax
    else:
        rvals = np.sort([*rvals, rho0_rmax])
        
    # For r>=r0, rhovals gives the offsets from the FQPM boundaries that we'll sample
    # Note: we sample these in both directions and we also always sample exactly on the FQPM bound and at offsets of 45deg. 
    rhomax = 2. * np.pi * rmax * (45./360.)
    if log_rhoscale:
        rhovals = _logspace_min_step(drmin, rhomax, nrho, drmin, dtype=np.float32)
    else:
        rhovals = np.linspace(drmin, rhomax, nrho, dtype=np.float32)

    rvals_all = [0.] # Always sample at exactly the coronagraph center
    thvals_all = [0.]

    field_rot = 0 if inst._rotation is None else inst._rotation

    for rval in rvals:
        if rval < rho0_rmax:
            dthetas = _dtheta_for_r_and_projsep(rval, rhovals[rhovals > rho0])
        else:
            dthetas = _dtheta_for_r_and_projsep(rval, rhovals)
            
        dthetas = dthetas[np.isfinite(dthetas) & (np.abs(dthetas)<=45.)]
        for th0 in np.array([0.,90.,180.,270.]):
            thvals = th0 + np.sort([0., 45., *(-dthetas), *(dthetas)])
            for th in thvals:
                rvals_all.append(rval)
                thvals_all.append(th-field_rot)

    rvals_all = np.array(rvals_all)
    thvals_all = np.array(thvals_all)

    c_coron0 = np.array([siaf_ap.XSciRef, siaf_ap.YSciRef])-1

    if c_coron_rolls is None:
        c_coron_rolls = np.array([c_coron0])

    # Generate coordinate maps for each roll
    rg_rolls, tg_rolls = [], []
    for c_coron in c_coron_rolls:
        ygpx, xgpx = c_to_c_osamp(np.indices((ny*osamp, nx*osamp), dtype=np.float32), 1/osamp)

        xg, yg = siaf_ap.sci_to_idl(xgpx - c_coron[0] + c_coron0[0], ygpx - c_coron[1] + c_coron0[1])

        rg, tg = webbpsf_ext.coords.xy_to_rtheta(xg, yg)
        rg = rg.round(5)

        rg_rolls.append(rg)
        tg_rolls.append(tg)

    rg_rolls, tg_rolls = np.array(rg_rolls), np.array(tg_rolls)

    tg_rolls = np.mod(tg_rolls, 360.)

    th_fqpm = np.array([0.0, 90.0, 180.0, 270.0]) - field_rot
    th_fqpm_wrap = np.array([*th_fqpm - 360.0, *th_fqpm, *th_fqpm + 360.0])
    th_fqpm_wrap0 = np.array([*th_fqpm, *th_fqpm, *th_fqpm])

    nearest_th_fqpm_rolls = th_fqpm_wrap0[np.argmin(np.abs(tg_rolls[None, :, :, :] - th_fqpm_wrap[:, None, None, None]), axis=0)]

    dthg = np.fmod(tg_rolls - nearest_th_fqpm_rolls, 360)
    dthg = np.where(dthg > 45., dthg - 360., dthg)

    rhog_rolls = 2. * np.pi * rg_rolls * (dthg/360.)

    r0_mask = np.logical_or((rg_rolls < r0), (np.abs(rhog_rolls) <= rho0)) & (rg_rolls < rho0_rmax) # Create a boolean array

    # Append our positions for r<r0
    rvals_all = np.array([*rvals_all, *(rg_rolls[r0_mask])])
    thvals_all = np.mod(np.array([*thvals_all, *((tg_rolls)[r0_mask])]), 360)

    # Eliminate duplicates if present
    rvals_all, thvals_all = np.unique(np.array([rvals_all, thvals_all]), axis=1)

    # Create a record array so we can do a two field sort quickly, to sort rvals_all and thvals_all by r>>theta
    A = np.zeros((len(rvals_all),), dtype=[('r', rvals_all.dtype), ('t', thvals_all.dtype)])
    A['r'], A['t'] = rvals_all, thvals_all
    order = np.argsort(A, order=['r', 't'])
    del A

    rvals_all, thvals_all = rvals_all[order], thvals_all[order]

    psf_offsets_polar = np.array([rvals_all, thvals_all])
    psf_offsets = np.array(webbpsf_ext.coords.rtheta_to_xy(*psf_offsets_polar))

    return psf_offsets, psf_offsets_polar


def _make_pixel_averaged_transmission_map(X,
                                         Y,
                                         T,
                                         xOut, 
                                         yOut,
                                         Nhalfsamp=2,
                                         method="clough",
                                         fill_value=np.nan):
    """
    Create a pixel-averaged transmission map from scattered measurements.

    Parameters
    ----------
    X, Y, T : array-like
        Matching arrays of source positions and mask transmission.
    x_edges, y_edges : array-like
        Pixel edges of the desired output grid.
        Output shape will be (len(y_edges)-1, len(x_edges)-1).
    oversamp : int
        Number of subpixel samples per axis per output pixel.
        Total samples per pixel = oversamp**2.
    method : {"linear", "clough"}
        Interpolation method.
    fill_value : float
        Value used outside the convex hull of the input points.

    Returns
    -------
    Tmap : ndarray
        Pixel-averaged transmission map with shape (ny, nx).
    """

    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()
    T = np.asarray(T).ravel()

    points = np.column_stack([X, Y])

    if method == "linear":
        interp = LinearNDInterpolator(points, T, fill_value=fill_value)
    elif method == "clough":
        interp = CloughTocher2DInterpolator(points, T, fill_value=fill_value)
    else:
        raise ValueError("method must be 'linear' or 'clough'")

    dXY = yOut[1,0] - yOut[0,0]
    dXYs = np.linspace(-dXY/2., dXY/2., 2*Nhalfsamp+3)[1:-1]
    
    dX, dY = np.meshgrid(dXYs, dXYs)
    xSamp = dX.flatten()[:, None, None] + xOut[None]
    ySamp = dY.flatten()[:, None, None] + yOut[None]
    
    tSamp = interp(xSamp, ySamp)
    tOut = tSamp.mean(axis=0)
    
    return tOut


def _distance_to_polar_vector(X, Y, T):
    """
    Compute the orthogonal distance from points (X, Y) to vectors extending
    from the origin at angle T.

    Parameters
    ----------
    X, Y : array_like
        Cartesian coordinates of points. Must be broadcast-compatible.
    T : array_like
        Polar angles in degrees, measured counterclockwise from +X. 
        Must be broadcast-compatible with X and Y.

    Returns
    -------
    dist : ndarray
        Distance from each point to the corresponding ray, with broadcasted
        shape of X, Y, and T.

    """
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    T = np.asarray(T)

    theta = np.deg2rad(T)

    ux = np.cos(theta)
    uy = np.sin(theta)

    # Projection of point onto ray direction
    proj = X * ux + Y * uy

    # Signed perpendicular distance to the infinite line
    # Positive/negative sign depends on coordinate convention.
    perp = X * uy - Y * ux

    return np.where(proj >= 0, perp, np.nan)


def _xy_from_rtheta_and_orthog_distance(R, T, D):
    """
    Convert true radial distance from the origin plus signed orthogonal
    distance from a PA ray into Cartesian coordinates.

    Assumes the math PA convention:
        T = 0 deg points along +X
        T = 90 deg points along +Y

    This uses the same signed-distance convention as:
        D = X * sin(theta) - Y * cos(theta)

    Parameters
    ----------
    R : array_like
        True radial distance of the point from the origin:
            R = sqrt(X**2 + Y**2)
    D : array_like
        Signed orthogonal distance from the polar vector defined by r=[0,inf], theta=T.
    T : array_like
        Position angle in degrees, math convention.

    Returns
    -------
    X, Y : ndarray
        Cartesian coordinates with broadcasted shape of R, D, and T.
    """

    R = np.asarray(R)
    D = np.asarray(D)
    T = np.asarray(T)

    R, D, T = np.broadcast_arrays(R, D, T)

    theta = np.deg2rad(T)

    ux = np.cos(theta)
    uy = np.sin(theta)

    # Perpendicular unit vector consistent with
    # D = X * sin(theta) - Y * cos(theta)
    px = uy
    py = -ux

    q2 = R**2 - D**2
    q2 = np.where(q2 >= 0, q2, np.nan)

    # Distance along the PA ray from the origin to the perpendicular projection.
    # Positive root assumes the point projects onto the forward ray.
    q = np.sqrt(q2)

    X = q * ux + D * px
    Y = q * uy + D * py

    return X, Y


def get_fqpm_transmission_map(inst_ext, c_coron, return_oversample=True, osamp=None, nd_squares=True, shape=None, posang=0,
                              c_star=None, interp_method='linear', interp_fill_value=1., interp_Nhalfsamp=5, nrExtrap=100):
     
    """
    Uses an array of FQPM transmission samples (sampled with STPSF) to generate
    an interpolated FQPM map of any dimension / oversampling.

    With nrExtrap > 0, the outermost radial samples are carefully extrapolated
    to fill in the FOV to the largest radius probed. This is reasonably
    accurate with the included samples, but care will be required if you're
    generating your own transmission samples.
    
    Note 1: the FQPM samples were derived for an oversampling factor of 2, so
    fidelity may suffer for osamp > 2.
    
    Note 2: transmission derived assuming a blackbody source of Teff = 80K.
    This could be generalized in the future but the impact is small in
    practical applications.

    Note 3: Transmission values here account for the difference in encircled
    energy for a finite aperture as a function of detector position, which
    changes substantially across the FOV for the FQPMs. This was done by
    computing a dense STPSF model grid with normalize='first', and then
    generating the same grid a 2nd time with normalize='exit_pupil'. The
    transmission for each position is then the ratio of the 'first' to
    'exit_pupil' PSF sums, divided by the overall FQPM transmission (0.6). 
    """
    from astropy.table import Table
    if shape is None:
        ny, nx = inst_ext.siaf_ap.YSciSize, inst_ext.siaf_ap.XSciSize
    else:
        ny, nx = shape

    if osamp is None:
        osamp = inst_ext.oversample

    elif osamp != inst_ext.oversample:
        inst_ext.oversample = osamp

    # The transmission data are provided as (x,y) idl coords with corresponding
    # transmission values. We should therefore require only one set of
    # measurements for each filter (since each is paired to a specific mask for
    # MIRI coron). Even if an atypical subarray is used, the values should
    # still be valid. 
    f_transmission = _RESOURCE_DIR / 'miri_fqpm_transmission' / f'mask_transmission_data_{inst_ext.filter}.ecsv'
    if not f_transmission.exists():
        raise FileNotFoundError(f"FQPM transmission data file not found for filter: {inst_ext.filter}.\nExpected at: {f_transmission}")
    
    tdat = Table.read(f_transmission)

    yOut, xOut = c_to_c_osamp(np.indices((ny*osamp, nx*osamp), dtype=tdat['xidl'].dtype), 1/osamp)
    
    if posang != 0:
        if c_star is None:
            raise ValueError("If posang is specified, 'c_star' must be set to the stellar position (as [x,y]) in 0-indexed SCI coords.")
        xOut, yOut = xy_polar_ang_displacement(xOut-c_star[0], yOut-c_star[1], -posang) + c_star[:,np.newaxis,np.newaxis]

    c_coron0 = np.array([inst_ext.siaf_ap.XSciRef, inst_ext.siaf_ap.YSciRef])

    ridl = np.array(np.hypot(tdat['xidl'], tdat['yidl'])).round(5)
    ridl_uni, ridl_counts = np.unique(ridl, return_counts=True)

    if (nrExtrap != 0) and np.any(ridl_counts >= 8):     
        ridl_uni = ridl_uni[ridl_counts >= 8]
        rmaxidl = ridl_uni.max()
        rmaxidl_mask = ridl == rmaxidl

        tr_rmax = np.array(tdat['transmission'])[rmaxidl_mask]
        xidl_rmax = np.array(tdat['xidl'])[rmaxidl_mask]
        yidl_rmax = np.array(tdat['yidl'])[rmaxidl_mask]

        ygidl, xgidl = inst_ext.siaf_ap.sci_to_idl(xOut-c_coron[0]+c_coron0[0], yOut-c_coron[1]+c_coron0[1])
        
        rmaxidl_fov = np.hypot(xgidl, ygidl).max()

        rgvals_extra = np.linspace(rmaxidl, rmaxidl_fov*np.sqrt(2), nrExtrap+1)[1:]

        field_rot = inst_ext._rotation

        th_fqpm = np.array([0.0, 90.0, 180.0, 270.0]) - field_rot

        rho_rmax = np.array([_distance_to_polar_vector(xidl_rmax, yidl_rmax, theta) for theta in th_fqpm])

        thfqpm_rmax = th_fqpm[np.nanargmin(np.abs(rho_rmax), axis=0)]

        rho_rmax = np.take_along_axis(rho_rmax, np.nanargmin(np.abs(rho_rmax), axis=0, keepdims=True), 0)[0]

        xidl_extra = np.zeros((nrExtrap, rho_rmax.shape[0]), dtype=xidl_rmax.dtype)
        yidl_extra = np.zeros((nrExtrap, rho_rmax.shape[0]), dtype=yidl_rmax.dtype)
        tr_extra = np.zeros((nrExtrap, rho_rmax.shape[0]), dtype=tr_rmax.dtype)

        for i in range(nrExtrap):
            xidl_extra[i], yidl_extra[i] = _xy_from_rtheta_and_orthog_distance(rgvals_extra[i], thfqpm_rmax, rho_rmax)
            tr_extra[i] = tr_rmax

        psf_offsets_sci = inst_ext.siaf_ap.idl_to_sci([*tdat['xidl'], *xidl_extra.ravel()], [*tdat['yidl'], *yidl_extra.ravel()]) - c_coron0[:, None] 
        psf_tr = np.array([*tdat['transmission'], *tr_extra.ravel()])

    else:
        psf_offsets_sci = inst_ext.siaf_ap.idl_to_sci(tdat['xidl'], tdat['yidl']) - c_coron0[:, None]
        psf_tr = tdat['transmission']

    psf_pos_sci = psf_offsets_sci + c_coron[:, None]

    im_mask_osamp = _make_pixel_averaged_transmission_map(*psf_pos_sci, 
                                                          psf_tr, 
                                                          xOut, 
                                                          yOut, 
                                                          Nhalfsamp=interp_Nhalfsamp, 
                                                          method=interp_method, 
                                                          fill_value=interp_fill_value
                                                          )

    # The comparable NIRCam / Lyot function returns undistorted transmission
    # maps. However, the maps here are effectively already distorted by our
    # mapping of idl >> sci coords. The better approach is to adjust the
    # coordinate samples above before passing into
    # _make_pixel_averaged_transmission_map. However, as a quick fix, we're
    # just distortion correcting the output here instead.
    im_mask_osamp, _ = undistort_image(im_mask_osamp,
                                       inst_ext.siaf_ap,
                                       c_coron,
                                       pxscale_out=inst_ext.pixelscale,
                                       osamp=osamp,
                                       method='linear',
                                       fill_value=0)

    if return_oversample:
        return im_mask_osamp

    im_mask = webbpsf_ext.image_manip.frebin(im_mask_osamp, scale=1/osamp, total=False)
    return im_mask