from .utils import (xy_polar_ang_displacement, c_to_c_osamp, pad_or_crop_image, px_size_to_ang_size, ang_size_to_proj_sep, ang_size_to_px_size, dist_to_pt)
from .distortion import get_siaf_aper
from .convolution import (psf_convolve_cpu, apply_partial_dist_to_stpsf_hdul, source_spectrum_to_weights)
from .rdi import (compute_rdi_coefficients)
from .space import SpaceReduction

import numpy as np
import astropy.units as u
from copy import deepcopy
from dataclasses import dataclass
import lmfit
from astropy.io import fits
from webbpsf_ext import image_manip
from astropy.convolution import Gaussian2DKernel
from vip_hci.fm import ScatteredLightDisk

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
            hdul = apply_partial_dist_to_stpsf_hdul(hdul, inst)
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

    if spacerdi.convolver.blursigma != 0:
        blur_kernel = Gaussian2DKernel(spacerdi.convolver.blursigma)

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

            if spacerdi.convolver.blursigma != 0: # This is applied to the data without oversampling, so is applied likewise here.
                im = psf_convolve_cpu(im, blur_kernel)

            hcube[i,j] = im
        
    hcube = (hcube*(u.mJy/u.pixel**2)/spacerdi.pxscale**2).to(u.MJy/u.sr).value
    
    if return_components:
        return hcube
    return np.nansum(hcube, axis=0)


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
        lwidth=100.0):
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

            cosphi = (rstar * s * np.sin(theta_star) + zd * c) / dstar_safe
            cosphi = np.clip(cosphi, -1.0, 1.0)

            r_disk = np.sqrt((xd - self.xdo) ** 2 + (yd - self.ydo) ** 2)
            theta_disk = np.arctan2(yd - self.ydo, xd - self.xdo)
            costheta = np.cos(theta_disk - omega_rad)

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                rho = self.dust_density.density_cylindrical(r_disk, costheta, zd)
                phase = self.phase_function.compute_phase_function_from_cosphi(cosphi)
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


###############################################################################
# Composite scene models
#
# A circumstellar scene is built from independent ModelComponent objects
# (rings, point sources, backgrounds, ...; write your own by subclassing)
# that are collected in a CompositeModel. Each component generates a
# PSF-convolved model cube matching the observations. The CompositeModel
# forward models every component through the current RDI procedure, solves
# (linearly) for the brightness of each component that isn't being fit
# explicitly, and uses lmfit to optimize all of the remaining parameters.
###############################################################################

class ModelGrid:
    """
    Geometry of the north-up grid on which a RawModelComponent renders its
    unconvolved model.

    Attributes
    ----------
    nx, ny : int
        Dimensions of the (possibly oversampled) grid.

    pxscale : astropy.units.Quantity
        Pixel scale of the grid (arcsec/pixel).

    center : numpy.ndarray
        [x, y] pixel position of the star on the grid.

    distance : float or None
        Distance to the target in parsecs.

    osamp : int
        Oversampling of the grid relative to the detector sampling.

    rmax_accuracy : float or None
        The largest separation (in au) at which the model needs to be accurate.
        Models can safely be truncated beyond this separation.
    """
    def __init__(self, nx, ny, pxscale, center, distance=None, osamp=1, rmax_accuracy=None):
        self.nx = int(nx)
        self.ny = int(ny)
        self.pxscale = pxscale
        self.center = np.asarray(center)
        self.distance = distance
        self.osamp = osamp
        self.rmax_accuracy = rmax_accuracy

    def coords(self):
        """
        Returns (dx, dy): arrays holding the offset of each pixel from the star
        in arcsec, in the north-up frame (positive dx is to the west of the
        star and positive dy is to the north of the star).
        """
        yy, xx = np.indices((self.ny, self.nx))
        scale = (self.pxscale << u.arcsec/u.pixel).value
        return (xx - self.center[0])*scale, (yy - self.center[1])*scale


@dataclass
class ModelContext:
    """
    Information that a CompositeModel passes to its components whenever they
    generate a model.

    Attributes
    ----------
    spacerdi : winnie.space.SpaceRDI
        The SpaceRDI object (with the current crop applied).

    distance : float or None
        Distance to the target in parsecs.

    siaf_aper : pysiaf aperture
        Aperture (with any distortion coefficients being varied applied) to use for the astrometric distortion.

    distorted_coords : bool
        If True, point source positions are interpreted as distorted (i.e., naively
        assuming a uniform pixel scale); if False as undistorted ('idl' coordinates).

    rmax_accuracy : float or None
        The largest separation (in au) at which disk models need to be accurate.

    ptsrc_ext : int
        The extension of the STPSF HDULists to use for point sources (0 when the distortion
        is applied separately for each trial; 2 when the nominal distortion is already applied).

    npad : int
        Padding (in pixels) applied to point source model cubes before distortion is applied.

    dist_key : tuple
        The values of any distortion coefficients being varied.

    verbose : bool
        Whether components should print info about slow set-up steps.
    """
    spacerdi: object
    distance: object
    siaf_aper: object
    distorted_coords: bool
    rmax_accuracy: object
    ptsrc_ext: int
    npad: int
    dist_key: tuple
    verbose: bool = True


def _parse_param_spec(spec, default_bounds=(-np.inf, np.inf), vary=None):
    """
    Converts a parameter specification to keyword arguments for
    lmfit.Parameters.add. A specification can be a dictionary of keyword
    arguments (e.g., dict(value=1, min=0, max=2, vary=False); used as is), a
    3-element sequence of [initial, min, max], or a single (initial) value.

    If vary is None, parameters specified as [initial, min, max] vary and those
    specified by a single value do not.
    """
    if isinstance(spec, dict):
        return dict(spec)
    if np.size(spec) == 3:
        value, vmin, vmax = [float(x) for x in spec]
        return dict(value=value, min=vmin, max=vmax, vary=(True if vary is None else vary))
    if np.size(spec) == 1:
        return dict(value=float(spec), min=default_bounds[0], max=default_bounds[1], vary=(False if vary is None else vary))
    raise ValueError(f"Parameter specifications must be a single value, an [initial, min, max] sequence, or a dict of lmfit.Parameters.add arguments; got {spec!r}")


class ModelComponent:
    """
    Base class for a component of a circumstellar scene model (see
    CompositeModel). A component declares its lmfit parameters and generates a
    model cube for the observations. 
    
    The overall brightness of every component is described by a coefficient. By
    default the coefficient is solved for linearly whenever the model is
    evaluated, and so is not a parameter that lmfit has to optimize ('autoscaled').
    Each component nevertheless has a '<label>_coeff' parameter, which is used
    instead whenever the component is not autoscaled. By default, a component is
    not autoscaled if its coefficient parameter is set to vary or to
    anything but 1. Alternatively, set the autoscale attribute to True or False.

    To write a component of your own, subclass ModelComponent (or
    RawModelComponent for anything that can be described as an unconvolved image
    of the sky) and:

      - set the 'kind' class attribute if you want labels to be assigned
        automatically (e.g., 'gal' -> 'gal1', 'gal2', ...) 
      - override param_specs to declare the component's parameters (excluding
        the coefficient)
      - override generate to return the model cube for the current parameter
        values (optionally with the unconvolved model as a tuple: (modelcube, raw_model))

    Parameters
    ----------
    label : str, optional
        Unique name for the component (used to label its products, and as the
        prefix for its parameters: '<label>_<name>'). If None, the label is assigned when
        the component is added to a CompositeModel, based on the 'kind'.

    autoscale : bool, optional
        If True, the coefficient is always solved for. If False, the
        coefficient is always taken from the '<label>_coeff' parameter. If None
        (default), the coefficient is solved for unless the '<label>_coeff' parameter
        varies, is tied to another parameter by an expression, or is not equal to 1.

    coeff : float, optional
        Initial value of the coefficient parameter. If provided (and autoscale
        is not True), the component is not autoscaled and the coefficient
        parameter varies.

    Attributes
    ----------
    kind : str or None
        Stem for automatically assigned labels.

    coeff_min : float
        Lower bound of the coefficient parameter.

    uses_distortion : bool
        Whether the component's model depends on the astrometric distortion
        solution (i.e., has to be regenerated when distortion coefficients vary).
    """
    kind = None
    coeff_min = -np.inf
    uses_distortion = True
    _scale_keys = ('coeff',) # parameters that, if explicitly varied/set, turn off autoscaling

    def __init__(self, label=None, autoscale=None, coeff=None):
        self.label = label
        self.coeff = coeff
        self.autoscale = (False if (coeff is not None and autoscale is None) else autoscale)

    def __repr__(self):
        return f"{type(self).__name__}(label={self.label!r})"

    def pname(self, key):
        """Name of this component's lmfit parameter for 'key'."""
        return f'{self.label}_{key}'

    def param_specs(self):
        """
        Returns a list of (key, kwargs) for each of the component's parameters
        (excluding the coefficient), where kwargs are the arguments passed to
        lmfit.Parameters.add. The parameter is named '<label>_<key>'.
        """
        return []

    @property
    def keys(self):
        """Keys of the component's parameters (excluding the coefficient)."""
        return [key for key, _ in self.param_specs()]

    def param_names(self):
        """Names of all of the component's lmfit parameters."""
        return [self.pname(key) for key in [*self.keys, 'coeff']]

    def add_params(self, params):
        """Adds the component's parameters to an lmfit.Parameters object."""
        for key, kws in self.param_specs():
            params.add(self.pname(key), **kws)
        explicit = (self.autoscale is False)
        params.add(self.pname('coeff'), value=(1. if self.coeff is None else self.coeff), min=self.coeff_min, max=np.inf, vary=explicit)

    def is_autoscaled(self, params):
        """Whether the component's coefficient is to be solved for (rather than taken from its parameter)."""
        if self.autoscale is not None:
            return bool(self.autoscale)
        for key in self._scale_keys:
            name = self.pname(key)
            if name in params:
                par = params[name]
                if par.vary or (par.expr is not None) or (par.value != 1):
                    return False
        return True

    def values(self, pdict):
        """Returns {key: value} for the component's parameters (excluding the coefficient) in the dictionary of parameter values pdict."""
        return {key: pdict[self.pname(key)] for key in self.keys}

    def cache_key(self, pdict):
        """
        Hashable summary of the parameter values the generated model depends on. A
        CompositeModel only regenerates (and forward models) a component's
        model when this changes.
        """
        return tuple(pdict[self.pname(key)] for key in self.keys)

    def prepare(self, ctx, params):
        """
        Called by CompositeModel.prepare_fit (and prepare_final) so that
        a component can do any slow one-time set-up (e.g., generating PSFs).
        """
        pass

    def generate(self, ctx, pdict):
        """
        Generates the component's model for the parameter values in pdict
        (a dictionary of all parameter values; see ModelComponent.values) 
        for a coefficient of 1.

        Parameters
        ----------
        ctx : ModelContext
            Information about the data / fitting set-up. ctx.spacerdi is the SpaceRDI object.

        pdict : dict
            Dictionary of the current value of every parameter in the model.

        Returns
        -------
        modelcube : numpy.ndarray
            PSF-convolved model of shape (number of exposures, ny, nx) matching
            ctx.spacerdi.imcube_sci (the pixels that are NaN in the data are masked automatically).

        raw_model : numpy.ndarray, optional
            Unconvolved north-up model at detector sampling, if the component
            has one. To include it, return the tuple (modelcube, raw_model);
            otherwise, just return modelcube.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement generate().")


class RawModelComponent(ModelComponent):
    """
    A component described by an unconvolved, north-up image of the sky. The
    image is rotated to each roll angle, convolved with the PSF grid of the
    SpaceRDI object (and distorted) by Winnie.
    
    Either subclass RawModelComponent and override make_raw_model (and,
    if needed, param_specs), or provide model_fn and params directly. For
    example, an elliptical gaussian source:

        def gaussian_source(values, grid):
            dx, dy = grid.coords() # arcsec from the star; +dx is west, +dy is north
            x, y = dx - values['dx0'], dy - values['dy0']
            return np.exp(-0.5*((x/values['sigx'])**2 + (y/values['sigy'])**2))

        blob = RawModelComponent('blob', model_fn=gaussian_source,
                                 params=dict(dx0=[2., 1., 3.], dy0=[-1., -2., 0.], sigx=[0.3, 0.05, 1.], sigy=[0.1, 0.05, 1.]))

    Parameters
    ----------
    label : str, optional
        See ModelComponent.

    model_fn : callable, optional
        Function model_fn(values, grid) returning the unconvolved model as a
        2D array of shape (grid.ny, grid.nx). 'values' is a dictionary of the
        component's current parameter values (keyed by the names provided in
        params) and 'grid' is a ModelGrid. Only relative values matter when the
        component is autoscaled.

    params : dict, optional
        The component's parameters as {name: specification}, where each
        specification is a single value (fixed parameter), a sequence of [initial,
        min, max] (varying parameter), or a dictionary of keyword arguments for 
        lmfit.Parameters.add.

    osamp : int
        Oversampling factor for the raw model (relative to the detector
        pixels). Values > 1 may be needed for especially compact structure.

    autoscale, coeff : 
        See ModelComponent.
    """
    def __init__(self, label=None, model_fn=None, params=None, osamp=1, autoscale=None, coeff=None):
        super().__init__(label=label, autoscale=autoscale, coeff=coeff)
        self.model_fn = model_fn
        self.params = ({} if params is None else dict(params))
        self.osamp = int(osamp)

    def param_specs(self):
        return [(key, _parse_param_spec(spec)) for key, spec in self.params.items()]

    def make_raw_model(self, values, grid):
        """
        Returns the unconvolved north-up model image (shape (grid.ny,
        grid.nx)) for the parameter values in 'values'.
        """
        if self.model_fn is None:
            raise NotImplementedError(f"{type(self).__name__} needs a model_fn or must override make_raw_model().")
        return self.model_fn(values, grid)

    def prepare(self, ctx, params):
        sp = ctx.spacerdi
        if sp.convolver is None or sp.convolver.psfs is None:
            raise ValueError(f"The model component '{self.label}' needs PSF convolution; run the prepare_convolution method of your SpaceRDI object first.")

    def generate(self, ctx, pdict):
        sp = ctx.spacerdi
        osamp = self.osamp
        grid = ModelGrid(sp.nx*osamp, sp.ny*osamp, sp.pxscale/osamp, c_to_c_osamp(sp.c_star, osamp),
                         distance=ctx.distance, osamp=osamp, rmax_accuracy=ctx.rmax_accuracy)
        raw = np.asarray(self.make_raw_model(self.values(pdict), grid))
        if raw.shape != (grid.ny, grid.nx):
            raise ValueError(f"The raw model for '{self.label}' has shape {raw.shape}, but should have shape {(grid.ny, grid.nx)}.")
        modelcube = sp.convolver.convolve_model(raw, pxscale_in=grid.pxscale, c_star_in=grid.center, distortion_aper=ctx.siaf_aper)
        if osamp != 1:
            raw = image_manip.frebin(raw, scale=1./osamp, total=False)
        return modelcube, raw

class RingComponent(RawModelComponent):
    """
    A ring-like disk component, rendered with grater_2hg_disk_model (a simple
    ring following Augereau et al. 1999, with a scattering phase function
    described by one or two Henyey-Greenstein terms).

    Each of the disk parameters below can be given as a single value (the
    initial value, with the default bounds) or as a sequence of [initial, min,
    max]. The defaults for the optional parameters and all of the default bounds
    are the same as for the arguments of grater_2hg_disk_model. The component's
    parameters are named '<label>_<parameter>' (e.g., 'ring1_r0').

    The component's coefficient is the scaling of the unconvolved disk model, 
    which is arbitrary unless the 'flux_max' option is used: with flux_max=1, 
    the model is normalized to a peak (unconvolved) surface brightness of 1
    before convolution, so the coefficient becomes the peak surface brightness
    in MJy/sr (at the sampling of the model, and so a strong function of how the
    disk aligns with the pixel grid). 

    Parameters
    ----------
    r0 : float or sequence
        Fiducial radius in au.

    pa, incl, ain, aout, h0, beta, gamma, e, omega : float or sequence, optional
        Position angle, inclination, inner and outer radial density exponents,
        ratio of scale height to r0 at r0, radial flaring exponent, vertical
        density exponent, eccentricity, and argument of pericenter. Initial values, 
        bounds, and which parameters vary (all but beta, gamma, e and omega by
        default) are described above and in the documentation of
        grater_2hg_disk_model.
        
    g1, g2, wg1 : float or sequence, optional
        Henyey-Greenstein asymmetry parameters and the weight of the g1 term
        in the scattering phase function (see 'spf').
    
    spf : {'twohg', 'onehg', 'isotropic'}
        'twohg': the scattering phase function is a linear combination of two
        Henyey-Greenstein SPFs, with varying asymmetry parameters (g1 and g2) and weight (wg1). 
        'onehg': a single Henyey-Greenstein SPF with a varying asymmetry parameter, g1.
        'isotropic': isotropic scattering, emulated by fixing the asymmetry parameter of a single
        Henyey-Greenstein SPF to 0.

    vary_beta, vary_gamma, vary_e : bool
        Whether the optional parameters beta, gamma, and e (and omega; set by
        vary_e) vary. All are False by default.
    
    coplanar_with : RingComponent or str, optional
        If provided, this ring's position angle and inclination are tied to
        those of the indicated ring (given as a component or its label), which
        must be added to the CompositeModel first.
        
    coeccentric_with : RingComponent or str, optional
        Likewise for eccentricity and argument of pericenter. 

    halfNbSlices : int
        The number of planar slices to compute above and below the disk
        midplane when generating the raw disk model.

    osamp : int
        Oversampling factor for the raw disk model. Values > 1 will increase
        runtimes but may be necessary for especially compact and/or narrow disks.

    rmax_accuracy : float, optional
        The largest separation (in au) at which to calculate the disk model. If
        None, uses the value set by the CompositeModel (by default, the edge of
        the field of view).

    xdo, ydo, flux_max : float or sequence, optional
        Additional parameters of grater_2hg_disk_model (offset of the disk
        center from the star in au and peak surface brightness). These are not included
        unless provided. Unlike the parameters above, these are fixed
        when specified as a single value and only vary if given as [initial, min, max].

    label, autoscale, coeff :
        See ModelComponent.
    """
    kind = 'ring'
    coeff_min = 0.
    _scale_keys = ('coeff', 'flux_max')
    _extra_keys = ('xdo', 'ydo', 'flux_max')

    def __init__(self, r0, pa=None, incl=None, ain=None, aout=None, h0=None, beta=None, gamma=None, e=None, omega=None,
                 g1=None, g2=None, wg1=None, spf='twohg', vary_beta=False, vary_gamma=False, vary_e=False,
                 coplanar_with=None, coeccentric_with=None, halfNbSlices=25, osamp=1, rmax_accuracy=None,
                 label=None, autoscale=None, coeff=None, **extra):
        if spf not in ['twohg', 'onehg', 'isotropic']:
            raise ValueError("spf must be 'twohg', 'onehg', or 'isotropic'")
        unknown = [key for key in extra if key not in self._extra_keys]
        if len(unknown) != 0:
            raise TypeError(f"Unexpected keyword argument(s) for RingComponent: {unknown}. Additional disk parameters can be any of {list(self._extra_keys)}.")
        super().__init__(label=label, osamp=osamp, autoscale=autoscale, coeff=coeff)
        self.disk_params = dict(r0=r0, pa=pa, incl=incl, ain=ain, aout=aout, h0=h0, beta=beta, gamma=gamma, e=e, omega=omega, g1=g1, g2=g2, wg1=wg1)
        self.extra_params = dict(extra)
        self.spf = spf
        self.vary_ring_pars = dict(r0=True, pa=True, incl=True, ain=True, aout=True, h0=True,
                                   beta=vary_beta, gamma=vary_gamma, e=vary_e, omega=vary_e)
        self.coplanar_with = coplanar_with
        self.coeccentric_with = coeccentric_with
        self.halfNbSlices = halfNbSlices
        self.rmax_accuracy = rmax_accuracy

    @staticmethod
    def _tied_label(ref):
        return (ref.label if isinstance(ref, ModelComponent) else ref)

    def _ties(self):
        ties = {}
        for ref, keys in [(self.coplanar_with, ('pa', 'incl')), (self.coeccentric_with, ('e', 'omega'))]:
            if ref is not None:
                ref_label = self._tied_label(ref)
                if ref_label is None:
                    raise ValueError("Rings can only be tied to components that have a label.")
                ties.update({key: f'{ref_label}_{key}' for key in keys})
        return ties

    def param_specs(self):
        ties = self._ties()
        pars = self.disk_params
        specs = []

        if np.size(pars['r0']) == 3:
            r0_spec = _parse_param_spec(pars['r0'], vary=self.vary_ring_pars['r0'])
        else:
            r0_spec = _parse_param_spec(pars['r0'], default_bounds=(1., 1000.), vary=self.vary_ring_pars['r0'])
        specs.append(('r0', r0_spec))

        for key, (v0, vmin, vmax) in _RING_PARAM_DEFAULTS.items():
            if key in ties:
                specs.append((key, dict(expr=ties[key])))
            elif pars[key] is None:
                specs.append((key, dict(value=v0, min=vmin, max=vmax, vary=self.vary_ring_pars[key])))
            else:
                specs.append((key, _parse_param_spec(pars[key], default_bounds=(vmin, vmax), vary=self.vary_ring_pars[key])))

        # Scattering parameters
        if self.spf == 'isotropic':
            g1_default = dict(value=0.0, min=-0.99, max=0.99, vary=False)
        else:
            g1_default = dict(value=0.8, min=0., max=0.99, vary=True)
        
        if self.spf == 'twohg':
            g2_default = dict(value=-0.2, min=-0.99, max=0.99, vary=True)
            wg1_default = dict(value=0.65, min=0.0, max=1.0, vary=True)
        else:
            g2_default = dict(value=0.0, min=-0.99, max=0.99, vary=False)
            wg1_default = dict(value=1.0, min=0.0, max=1.0, vary=False)

        for key, default in [('g1', g1_default), ('g2', g2_default), ('wg1', wg1_default)]:
            if pars[key] is None:
                specs.append((key, default))
            else:
                specs.append((key, _parse_param_spec(pars[key], default_bounds=(default['min'], default['max']), vary=default['vary'])))

        for key, spec in self.extra_params.items():
            specs.append((key, _parse_param_spec(spec)))

        return specs

    def add_params(self, params):
        for key, expr in self._ties().items():
            if expr not in params:
                raise ValueError(f"'{self.label}' is tied to a ring that hasn't been added to the model yet: parameter '{expr}' not found.")
        super().add_params(params)

    def make_raw_model(self, values, grid):
        rmax_accuracy = (grid.rmax_accuracy if self.rmax_accuracy is None else self.rmax_accuracy)
        # Unless it is normalized to a peak value (flux_max), the raw model scales with pixel area; so we
        # correct for oversampling to keep a consistent normalization
        scale = (1. if values.get('flux_max') is not None else self.osamp**2)
        return scale * grater_2hg_disk_model(**values, cent=grid.center, distance=grid.distance, nx=grid.nx, ny=grid.ny,
                                             pxscale=grid.pxscale, rmax_accuracy=rmax_accuracy, halfNbSlices=self.halfNbSlices)

    def prepare(self, ctx, params):
        super().prepare(ctx, params)
        if ctx.distance is None:
            raise ValueError(f"A distance is needed to model the ring '{self.label}'; set the distance for your CompositeModel.")


class PointSourceComponent(ModelComponent):
    """
    A point-like source: e.g., a planet or other companion, a background star,
    or a background galaxy (which can be marginally resolved when blurring is
    included). The model is generated with STPSF PSFs that account for the
    coronagraph (and, optionally, for any additional blurring), and positioned
    in each roll based on the source's position in the north-up frame.
    
    The component's parameters are named '<label>_<parameter>' (e.g.,
    'ptsrc1_dx'). Its coefficient is the flux in mJy. The initial STPSF PSFs
    are generated at the initial position of the source (when the
    CompositeModel is prepared, which can take a while) and are then simply
    shifted for the other positions that are tried. For sources close to the
    coronagraph IWA, it is therefore important that the initial position is
    close to the truth. If it isn't, set the initial position to the result
    of a first fit and refresh the PSFs (see reset_psfs). 

    Parameters
    ----------
    dx, dy : float
        Initial position (in arcsec) of the source relative to the star, in the
        north-up frame: positive dx is west of the star and positive dy is north
        of the star.

    flux : float, optional
        Initial flux of the source in mJy. If provided, the flux is fit
        directly (rather than solved for linearly; see ModelComponent).

    dxy_range : float
        The maximum offset (in arcsec) of the source from the initial
        position in each axis.

    det_blur, sky_blur : {None, 'gaussian', 'gaussian2d'}
        Optional additional blurring of the source, with a gaussian kernel
        having standard deviations in detector pixels. det_blur emulates extra
        detector effects (e.g., BFE) for sources that are truly point-like,
        with the kernel oriented relative to the detector frame. sky_blur
        emulates slightly extended sources (e.g., marginally resolved background
        galaxies), with the kernel oriented relative to the north-up sky frame.
        If None, no blurring is included. If 'gaussian', the kernel is
        circular with a single varying sigma ('<label>_detxsig' or
        '<label>_skyxsig'). If 'gaussian2d', separate x and y sigmas
        ('...xsig', '...ysig') and an angle theta in degrees ('...theta')
        vary. The sigmas are initialized at a negligible value, so you will
        generally want to set better initial values.

    spectrum : synphot spectrum or dict, optional
        Spectrum of the source, used to generate the PSFs. Either a synphot
        spectrum, or a dictionary with entries 'wavelengths' (in meters) and
        'weights' (see winnie.convolution.source_spectrum_to_weights). Defaults to an M5V
        spectrum (as in generate_ptsrc_hduls).

    fov_pixels : int
        Field of view of the generated PSFs in detector pixels. For bright
        sources whose PSF wings are detected to large separations, this should be larger.

    osamp : int
        Oversampling of the generated PSFs.

    nlambda : int, optional
        Number of wavelengths for the PSFs. By default, this matches the
        convolver's setting; for wide filters and bright sources, nlambda ~ 50
        should work across the FOV.

    normalize : str
        Normalization for the STPSF PSFs.

    label, autoscale :
        See ModelComponent.
    """
    kind = 'ptsrc'
    coeff_min = 0.

    def __init__(self, dx, dy, flux=None, dxy_range=0.5, det_blur=None, sky_blur=None, spectrum=None,
                 fov_pixels=151, osamp=2, nlambda=None, normalize='exit_pupil', label=None, autoscale=None):
        if det_blur not in [None, 'gaussian', 'gaussian2d']:
            raise ValueError("det_blur must be None, 'gaussian', or 'gaussian2d'")
        if sky_blur not in [None, 'gaussian', 'gaussian2d']:
            raise ValueError("sky_blur must be None, 'gaussian', or 'gaussian2d'")
        super().__init__(label=label, autoscale=autoscale, coeff=flux)
        self.dx = dx
        self.dy = dy
        self.dxy_range = dxy_range
        self.det_blur = det_blur
        self.sky_blur = sky_blur
        self.spectrum = spectrum
        self.fov_pixels = fov_pixels
        self.osamp = osamp
        self.nlambda = nlambda
        self.normalize = normalize
        self._hduls = None
        self._hduls_key = None
        self._psf_shift = None

    def param_specs(self):
        specs = [('dx', dict(value=self.dx, min=self.dx-self.dxy_range, max=self.dx+self.dxy_range, vary=True)),
                 ('dy', dict(value=self.dy, min=self.dy-self.dxy_range, max=self.dy+self.dxy_range, vary=True))]
        for prefix, blur in [('det', self.det_blur), ('sky', self.sky_blur)]:
            if blur is None:
                continue
            specs.append((f'{prefix}xsig', dict(value=1e-10, min=1e-10, max=10, vary=True)))
            if blur == 'gaussian':
                specs.append((f'{prefix}ysig', dict(expr=self.pname(f'{prefix}xsig'))))
                specs.append((f'{prefix}theta', dict(value=0.0, vary=False)))
            else:
                specs.append((f'{prefix}ysig', dict(value=1e-10, min=1e-10, max=10, vary=True)))
                specs.append((f'{prefix}theta', dict(value=0.0, min=-180, max=180, vary=True)))
        return specs

    def reset_psfs(self):
        """
        Discards the generated PSFs so that they are regenerated (at the
        current initial position of the source) the next time the CompositeModel
        is prepared. Needed if you change the source's spectrum (or want the
        PSFs to be centered on a new position).
        """
        self._hduls = None
        self._hduls_key = None

    def prepare(self, ctx, params):
        sp = ctx.spacerdi
        conv = sp.convolver
        if conv is None:
            raise ValueError(f"The point source '{self.label}' needs a convolver; run the prepare_convolution method of your SpaceRDI object first.")

        key = (sp.concat, ctx.distorted_coords, self.fov_pixels, self.osamp, self.nlambda, self.normalize)
        if self._hduls is not None and self._hduls_key == key:
            return

        spectrum = (stpsf.specFromSpectralType('M5V', catalog='ck04') if self.spectrum is None else self.spectrum)
        if isinstance(spectrum, dict):
            weights = spectrum
            nlambda = len(weights['weights'])
            conv.calc_psf_shift(source_spectrum=weights, nlambda=nlambda)
        else:
            conv.calc_psf_shift(source_spectrum=spectrum, nlambda=self.nlambda)
            nlambda = conv._psfcube_off.shape[0]
            weights = source_spectrum_to_weights(spectrum, conv.inst_webbpsfext.bandpass, nlambda=nlambda)

        if ctx.verbose:
            print(f"Generating PSFs for '{self.label}' (one for each roll); this may take a moment...")
        position = [dict(dx=params[self.pname('dx')].value, dy=params[self.pname('dy')].value)]
        hduls = generate_ptsrc_hduls(sp, position, spectrum=weights, fov_pixels=self.fov_pixels, osamp=self.osamp,
                                     distorted_coords=ctx.distorted_coords, nlambda=nlambda, normalize=self.normalize)
        self._hduls = [roll_hduls[0] for roll_hduls in hduls]
        self._psf_shift = conv._psf_shift
        self._hduls_key = key

    def generate(self, ctx, pdict):
        if self._hduls is None:
            raise RuntimeError(f"The PSFs for '{self.label}' have not been generated; run prepare_fit (or prepare_final) for your CompositeModel first.")
        sp = ctx.spacerdi
        values = self.values(pdict)
        ptsrc_params = [dict(dx=values['dx'], dy=values['dy'], flux=1.)]
        blur_params = {f'{key}_1': val for key, val in values.items() if key not in ['dx', 'dy']}
        sp.convolver._psf_shift = self._psf_shift # the shift is specific to the source's spectrum
        modelcube = generate_nptsrc_model_cube(sp, ptsrc_params, ptsrc_hduls=[[hdul] for hdul in self._hduls], return_components=True,
                                               distorted_coords=ctx.distorted_coords, siaf_aper=ctx.siaf_aper, npad=ctx.npad,
                                               ext=ctx.ptsrc_ext, blur_params=blur_params)[0]
        if ctx.npad != 0:
            modelcube = modelcube[:, ctx.npad:-ctx.npad, ctx.npad:-ctx.npad]
        return modelcube, None


class UniformBackground(ModelComponent):
    """
    A uniform background offset (e.g., a sky or bias offset that was
    over/under-subtracted before the data reached Winnie). The model is a
    uniform image that is neither convolved nor attenuated by the
    coronagraph, so the coefficient is the surface brightness of the offset in
    MJy/sr (which can be negative).

    Parameters
    ----------
    label, autoscale, coeff :
        See ModelComponent.
    """
    kind = 'bg'
    uses_distortion = False

    def __init__(self, label='bg', autoscale=None, coeff=None):
        super().__init__(label=label, autoscale=autoscale, coeff=coeff)

    def generate(self, ctx, pdict):
        return np.ones_like(ctx.spacerdi.imcube_sci), None


class ModelCubeComponent(ModelComponent):
    """
    A component with a fixed, precomputed model for which only the brightness
    is solved for or fit. E.g., an already convolved model from another code
    or a previous fit.

    Parameters
    ----------
    label : str
        See ModelComponent.

    modelcube : numpy.ndarray
        The PSF-convolved model, in units of MJy/sr, with the roll angles of
        the observations and the shape of the full frame of the SpaceRDI
        object (i.e., the shape of spacerdi.imcube_sci when no crop is set,
        including any padding). It is cropped to match the data when needed.
        (A cube that already has the cropped shape is accepted while the data
        are cropped, but can't be used for finalize, which uses the uncropped
        data.)

    autoscale, coeff :
        See ModelComponent.
    """
    uses_distortion = False

    def __init__(self, label, modelcube, autoscale=None, coeff=None):
        super().__init__(label=label, autoscale=autoscale, coeff=coeff)
        self.modelcube = np.asarray(modelcube)

    def generate(self, ctx, pdict):
        sp = ctx.spacerdi
        modelcube = self.modelcube
        if modelcube.shape[-2:] != (sp.ny, sp.nx):
            y1, y2, x1, x2 = sp._crop_indices
            modelcube = modelcube[..., y1:y2, x1:x2]
        if modelcube.shape != sp.imcube_sci.shape:
            raise ValueError(f"The model cube for '{self.label}' has shape {self.modelcube.shape}, which does not match the data (shape {sp.imcube_sci.shape} with the current crop; "
                             f"the full frame has shape {sp._imcube_sci.shape}).")
        return modelcube, None


class ModelSolution:
    """
    The products of evaluating a CompositeModel for a set of parameter values
    (see CompositeModel.evaluate and CompositeModel.finalize). The dictionaries
    below are keyed by the labels of the model components (in the order the
    components were added).

    Attributes
    ----------
    params : lmfit.Parameters
        The parameter values that were evaluated.

    labels : list of str
        Labels of the components.

    coeffs : dict
        The coefficient (brightness) of each component. The 
        unit-coefficient model of each component is scaled by this value:
        e.g., it is the flux in mJy for a PointSourceComponent and the 
        surface brightness in MJy/sr for a UniformBackground.

    autoscale : dict
        Whether each component's coefficient was solved for (True) or taken from its
        parameter (False).

    modelcubes : dict
        The PSF-convolved model cube of each component, scaled by its coefficient.

    fmreducs : dict
        The forward modeled reduction (SpaceReduction) of each component, scaled by its coefficient.

    raw_models : dict
        The unconvolved north-up model (at detector sampling) of each component that has
        one (e.g., rings), scaled by its coefficient.

    modelcube_composite : numpy.ndarray
        Sum of all components' model cubes.

    fmrdi_reduc : winnie.space.SpaceReduction
        The forward modeled reduction of the composite model.

    rdi_reduc : winnie.space.SpaceReduction
        The reduction of the data that the model was compared to.

    roi : numpy.ndarray
        The boolean region of interest mask used for the comparison.

    settings : dict
        Info about the optimization settings (q_clip and err_weighting).

    modelreducs, model_reduc, raw_reduc :
        Reductions (SpaceReduction) of the PSF-convolved model without forward
        modeling for each component, for the composite, and for the sum of the unconvolved
        models. These are only available if the solution was created by
        CompositeModel.finalize (and are None otherwise).
    """
    def __init__(self, params, labels, coeffs, autoscale, modelcubes, fmreducs, raw_models,
                 modelcube_composite, fmrdi_reduc, rdi_reduc, roi, settings):
        self.params = params
        self.labels = labels
        self.coeffs = coeffs
        self.autoscale = autoscale
        self.modelcubes = modelcubes
        self.fmreducs = fmreducs
        self.raw_models = raw_models
        self.modelcube_composite = modelcube_composite
        self.fmrdi_reduc = fmrdi_reduc
        self.rdi_reduc = rdi_reduc
        self.roi = roi
        self.settings = settings
        self.modelreducs = None
        self.model_reduc = None
        self.raw_reduc = None

    def __repr__(self):
        coeffs = ', '.join(f'{label}: {coeff:0.4g}' for label, coeff in self.coeffs.items())
        return f'ModelSolution(coeffs={{{coeffs}}})'

    @property
    def raw_model(self):
        """Sum of the unconvolved models of all components that have one (None if there are none)."""
        if len(self.raw_models) == 0:
            return None
        return np.nansum(list(self.raw_models.values()), axis=0)

    def model_dict(self):
        """
        Dictionary summarizing the solution and the optimization settings, for
        storing in FITS headers: the value of every parameter, the coefficient
        of every component (as '<label>_coeff'), q_clip, and err_weighting.
        """
        model_dict = self.params.valuesdict()
        model_dict.update({f'{label}_coeff': coeff for label, coeff in self.coeffs.items()})
        model_dict.update(self.settings)
        return model_dict


class CompositeModel:
    """
    A circumstellar scene model composed of independent components (see
    ModelComponent), together with the machinery to optimize its parameters by
    forward modeling RDI. The approach is that of Winnie's composite modeling
    tutorial: each component is forward modeled through the RDI procedure, 
    the brightness of every component that isn't being explicitly fit is solved
    for linearly (within the region of interest; see ModelComponent), and
    lmfit varies the remaining parameters to minimize the residuals.

    A typical workflow, for a SpaceRDI object 'wdb' with a loaded concatenation
    and a prepared convolver:

        model = CompositeModel(wdb, distance, [RingComponent(r0=32., incl=89.),
                                               PointSourceComponent(dx=-1.06, dy=-3.22),
                                               UniformBackground()])
        p = model.params # an lmfit.Parameters object; adjust it as you like (e.g., p['ring1_r0'].vary = False)

        model.prepare_fit(roi=dict(rmax=5.5, units=u.arcsec)) # crops the data, runs RDI on it, and defines the region of interest
        sol = model.evaluate(p) # a ModelSolution for the initial parameters
        res = model.fit(p) # optimizes the parameters (lmfit.minimize; Powell's method by default)
        final = model.finalize(res.params) # carries out the full-frame, derotated, distortion-corrected reduction and model for the best-fit solution (and saves the products)

    Components can be added with add_component (or in the call above). Each
    adds its parameters to model.params. To use a different optimizer,
    model.residuals can be passed to lmfit (e.g., lmfit.minimize) directly.

    Note that the model modifies the state of the SpaceRDI object: the crop and
    RDI presets are changed (see prepare_fit and finalize), and its circumstellar
    model is overwritten (by each component in turn while fitting, and by the
    composite model after evaluate or finalize, so that MCRDI can follow
    directly after one of those). If you change the settings of the SpaceRDI
    object afterwards (e.g., by switching to the MCRDI presets), run prepare_fit again 
    before evaluating or fitting the model again.

    Parameters
    ----------
    spacerdi : winnie.space.SpaceRDI
        SpaceRDI object with a loaded concatenation and (for any components that
        require convolution) a prepared convolver.

    distance : float, optional
        Distance to the target in parsecs (needed for disk components).

    components : list of ModelComponent, optional
        Components to add to the model.

    distorted_coords : bool
        If True, point source positions are interpreted as distorted (i.e.,
        north-up stellocentric offsets that naively assume a uniform pixel scale of
        spacerdi.pxscale). If False, positions are interpreted as undistorted
        ('idl' coordinates in SIAF terms).

    vary_dist_coeffs : bool or list of str
        If True, will include a subset of astrometric distortion parameters
        (Sci2IdlX10, Sci2IdlY10, Sci2IdlY11), initialized based on the
        pysiaf values for the observations. If a list of coefficient names
        (e.g., ['Sci2IdlX10']), only those coefficients are included.

    err_weighting : bool, numpy.ndarray, or callable
        If True, the residuals are weighted by the uncertainties of the data
        reduction (rdi_reduc.err_rolls). If False, residuals are not weighted.
        If an array matching the shape of rdi_reduc.rolls, those values are used
        instead. If a callable, it is called with the data reduction
        (SpaceReduction) and should return an array of the weights (useful, e.g.,
        to fix bad error values for both the fit and the final reduction).

    q_clip : tuple or list or numpy.ndarray
        If not None, q_clip gives a lower and upper quantile bound for the
        residuals. Any values outside the quantile range are clipped when
        evaluating goodness of fit. Can be useful for data with significant
        artifacts within the region of interest (e.g., poor reference match,
        uncorrected hot pixels, etc). E.g., q_clip = [5,95] will compute
        goodness of fit using only the inner 5th-95%ile of the distribution of
        residual pixel values in your region of interest.

    add_fm_eps : bool
        If True, adds a negligible random noise term to each image being forward
        modeled. Avoids a crash in the RDI coefficient calculation when the
        model is exactly zero across the entire optimization zone.

    verbose : bool
        If True, prints a few messages about the progress of slow set-up steps.

    Attributes
    ----------
    components : list of ModelComponent

    params : lmfit.Parameters
        The initial parameters, which you can modify in place or replace. 

    rdi_reduc, roi, rmax_accuracy :
        The reduction of the data, the boolean region of interest mask, and the
        largest separation (in au) at which disk models are calculated; all
        set by prepare_fit and prepare_final.
    """
    def __init__(self, spacerdi, distance=None, components=(), distorted_coords=True, vary_dist_coeffs=False,
                 err_weighting=True, q_clip=None, add_fm_eps=True, verbose=True):
        self.spacerdi = spacerdi
        self.distance = distance
        self.distorted_coords = distorted_coords
        self.err_weighting = err_weighting
        self.q_clip = q_clip
        self.add_fm_eps = add_fm_eps
        self.verbose = verbose
        self.components = []
        self.params = lmfit.Parameters()
        self.rdi_reduc = None
        self.roi = None
        self.rmax_accuracy = None
        self._roi_spec = None
        self._sigma = None
        self._state = None
        self._cache = {}
        self._siaf_memo = (None, None)
        self._dist_keys = []
        self._init_dist_params(vary_dist_coeffs)
        for component in components:
            self.add_component(component)

    def __repr__(self):
        nvary = sum(par.vary for par in self.params.values())
        return f'CompositeModel(components={self.labels}, n_varying_params={nvary})'

    @property
    def labels(self):
        """Labels of the model's components."""
        return [component.label for component in self.components]

    def get_component(self, label):
        """Returns the component with the indicated label."""
        for component in self.components:
            if component.label == label:
                return component
        raise KeyError(f"No component labeled '{label}'; the model's components are: {self.labels}")

    def add_component(self, component):
        """
        Adds a component to the model (and its parameters to self.params). If
        the component doesn't have a label yet, one is assigned (e.g., 'ring1',
        'ring2', 'ptsrc1', ...). Returns the component.
        """
        if not isinstance(component, ModelComponent):
            raise TypeError("Components must be ModelComponent objects.")
        if component.label is None:
            if component.kind is None:
                raise ValueError(f"{type(component).__name__} has no default label; please provide a label for the component.")
            n = 1
            while f'{component.kind}{n}' in self.labels:
                n += 1
            component.label = f'{component.kind}{n}'
        if not str(component.label).isidentifier():
            raise ValueError(f"Component labels are used in parameter names and so must be valid identifiers; got '{component.label}'.")
        if component.label in self.labels:
            raise ValueError(f"A component labeled '{component.label}' is already in the model.")
        clashes = [name for name in component.param_names() if name in self.params]
        if len(clashes) != 0:
            raise ValueError(f"The parameter name(s) {clashes} of '{component.label}' are already in use.")
        try:
            component.add_params(self.params)
        except Exception:
            for name in component.param_names(): # don't leave a partially added component behind
                self.params.pop(name, None)
            raise
        self.components.append(component)
        self.clear_cache()
        return component

    def clear_cache(self):
        """
        Clears the models stored for each component. By default, a component's model is only regenerated
        (and forward modeled) when the values of its parameters change. Use this if you change a
        component in some other way (e.g., altering an attribute).
        """
        self._cache = {}
        self._siaf_memo = (None, None)

    def _init_dist_params(self, vary_dist_coeffs):
        if isinstance(vary_dist_coeffs, bool):
            varying_dist_coeffs = (['Sci2IdlX10', 'Sci2IdlY10', 'Sci2IdlY11'] if vary_dist_coeffs else [])
        else: # assume it's a list of coeff names
            varying_dist_coeffs = list(vary_dist_coeffs)

        if len(varying_dist_coeffs) == 0:
            return
        if getattr(self.spacerdi, 'siaf_aper', None) is None:
            raise ValueError("To vary distortion coefficients, a concatenation must be loaded in your SpaceRDI object before creating the CompositeModel.")
        siaf_aper = self.spacerdi.siaf_aper
        for par in varying_dist_coeffs:
            value = siaf_aper.__dict__[par]
            if value == 0:
                print(f'Warning: {par} is zero in PySIAF! Varying this parameter may produce problematic results!')
            if par in ['Sci2IdlX10', 'Sci2IdlY11']:
                self.params.add(par, value=value, vary=True, min=value*0.95, max=value*1.05)
            else:
                self.params.add(par, value=value, vary=True, min=-0.1, max=0.1)
            self._dist_keys.append(par)

    ###########################################################################
    # Set-up
    ###########################################################################

    def prepare_fit(self, roi, crop='roi', crop_pad_nfwhm=5, rmax_accuracy='auto', rdi_presets=True, refresh_psfs=False):
        """
        Prepares the data for optimizing the model: sets the standard RDI presets
        and crops the data (which speeds up forward modeling), runs RDI on the
        cropped data without derotation or distortion correction (so that the
        forward modeled images can be compared to high fidelity data with fewer
        operations), defines the region of interest, and lets each component
        carry out any slow one-time set-up (e.g., generating the PSFs for
        point sources). 

        Parameters
        ----------
        roi : dict or callable
            Defines the region of interest (ROI): the pixels that are
            included when evaluating goodness of fit. Either a dictionary of
            keyword arguments for winnie.modeling.make_region_of_interest_mask
            (rmin, rmax, dxy_northup, rexcl_northup, dxy_detector,
            rexcl_detector, and units; see that function's documentation) or
            a function roi(reduc, spacerdi) that returns the boolean mask (of the
            same shape as the images in the SpaceReduction 'reduc') and that will
            be used for any reductions of the data.
            To a reasonable approximation, everything that isn't masked should
            be either the flux you're modeling, oversubtraction artifacts, or noise.

        crop : {'roi', 'auto', None} or array-like
            The cropped shape [ny, nx] to use during optimization, which
            should be large enough to include everything you are modeling plus
            some padding for convolution. If 'roi', the crop is set to include the
            ROI (given by roi['rmax']) plus crop_pad_nfwhm times the FWHM. If
            'auto', uses the SpaceRDI.set_crop option of the same name (the
            largest separation in the optimization zones plus padding). If None,
            the data are not cropped. The crop is centered on the star; where
            that would run past the edge of the data (e.g., for a star that is off-center
            in the frame), the crop is reduced to fit.

        crop_pad_nfwhm : float
            Padding (in units of the FWHM) added to the ROI when crop='roi'.

        rmax_accuracy : float, None, or 'auto'
            The largest separation (in au) at which to calculate disk models.
            If 'auto', the calculation is truncated at about the edge of the
            cropped field of view (or is not truncated when the data aren't cropped). 

        rdi_presets : bool
            If True, sets the RDI presets of the SpaceRDI object to those of a
            standard RDI reduction (rdi_presets). Otherwise, the current
            settings are used (e.g., to fit data that were reduced with
            HPFRDI settings).

        refresh_psfs : bool
            If True, discards any PSFs that were already generated for
            components and generates them again, using the current initial
            parameters (see PointSourceComponent).
        """
        sp = self.spacerdi
        self._require_concat()
        self._check_params(self.params)
        self._roi_spec = roi

        if refresh_psfs:
            for component in self.components:
                if hasattr(component, 'reset_psfs'):
                    component.reset_psfs()

        if rdi_presets:
            sp.rdi_presets()

        crop_shape = self._resolve_crop(crop, roi, crop_pad_nfwhm)
        sp.set_crop(crop_shape)
        if crop_shape is not None:
            requested = [int(n) for n in sp.cropped_shape] # (this resolves crop='auto')
            feasible = self._feasible_crop_shape(requested)
            if feasible != requested:
                if self.verbose:
                    print(f'The requested crop {requested} runs past the edge of the data for the position of the star; cropping to {feasible} instead.')
                sp.set_crop(feasible)
        rdi_reduc = sp.run_rdi(save_products=False, derotate=False, correct_distortion=False)

        if isinstance(rmax_accuracy, str) and rmax_accuracy == 'auto':
            if (self.distance is None) or (crop_shape is None):
                rmax_accuracy = None
            else:
                half_width = np.max(sp.imcube_sci.shape[-2:])/2.
                rmax_accuracy = ang_size_to_proj_sep(px_size_to_ang_size(half_width, sp.pxscale), self.distance).value
        self.rmax_accuracy = rmax_accuracy

        self._set_data(rdi_reduc)
        self._prepare_components()

        if self.verbose:
            nvary = sum(par.vary for par in self.params.values())
            print(f'Ready to fit: components {self.labels}; {nvary} varying parameters; '
                  f'data shape {tuple(sp.imcube_sci.shape[-2:])} with {int(np.sum(self.roi))} pixels in the ROI.')

    def prepare_final(self, roi=None, rmax_accuracy=None, rdi_presets=True, save_rdi=False):
        """
        Prepares for the final evaluation of the model: undoes any cropping, runs
        RDI on the data with derotation and distortion correction, and defines the
        region of interest for the result. This is done for you by finalize.

        Parameters
        ----------
        roi : dict or callable, optional
            The region of interest definition (see prepare_fit). Defaults to the
            one that was used by prepare_fit.

        rmax_accuracy : float, optional
            The largest separation (in au) at which to calculate disk models.
            Defaults to None (the edge of the field of view).

        rdi_presets : bool
            See prepare_fit.

        save_rdi : bool
            If True, saves the RDI reduction of the data.
        """
        sp = self.spacerdi
        self._require_concat()
        if roi is not None:
            self._roi_spec = roi
        if self._roi_spec is None:
            raise ValueError("The region of interest is not defined; either run prepare_fit first or provide roi.")

        sp.set_crop()
        if rdi_presets:
            sp.rdi_presets()
        rdi_reduc = sp.run_rdi(save_products=save_rdi)

        self.rmax_accuracy = rmax_accuracy
        self._set_data(rdi_reduc)
        self._prepare_components()

    def _require_concat(self):
        if self.spacerdi.concat is None:
            raise ValueError("A concatenation must be loaded in your SpaceRDI object (see load_concat).")

    def _check_params(self, params):
        missing = [name for component in self.components for name in component.param_names() if name not in params]
        missing += [name for name in self._dist_keys if name not in params]
        if len(missing) != 0:
            raise ValueError(f"The parameters are missing the following entries needed by the model's components: {missing}")
        if len(self.components) == 0:
            raise ValueError("The model has no components; add some with add_component.")

    def _resolve_crop(self, crop, roi, crop_pad_nfwhm):
        sp = self.spacerdi
        if crop is None:
            return None
        if isinstance(crop, str):
            if crop == 'auto':
                return 'auto'
            if crop != 'roi':
                raise ValueError("crop must be 'roi', 'auto', None, or a shape [ny, nx].")
            if callable(roi):
                raise ValueError("The size for the crop can't be inferred from a roi function; provide crop='auto', None, or a shape [ny, nx].")
            rmax = roi.get('rmax', np.inf)
            if not np.isfinite(rmax):
                return None
            units = roi.get('units', u.pix)
            rmax_px = (rmax if units == u.pix else ang_size_to_px_size(rmax*units, sp.pxscale).value)
            npx_crop = int(2*(np.ceil(rmax_px + crop_pad_nfwhm*sp._fwhm)))
            return [npx_crop, npx_crop]
        return [int(n) for n in np.asarray(crop)]

    def _feasible_crop_shape(self, shape):
        """
        SpaceRDI.set_crop centers the crop on the star, but where the cropped box would run past the edge
        of the frame (e.g., for a star that is off-center in the frame) the data are silently truncated
        while the shape of the crop is not updated. Returns the shape [ny, nx], no larger than the
        requested shape, for which the crop fits within the frame.
        """
        sp = self.spacerdi
        feasible = []
        for n, c, N in [(shape[0], sp._c_star[1], sp._ny), (shape[1], sp._c_star[0], sp._nx)]:
            n = min(int(n), N)
            while max(0, int(np.round(c - (n - 1.)/2.))) + n > N: # as in winnie.utils.crop_data
                n -= 1
            feasible.append(n)
        return feasible

    def _make_roi(self, reduc):
        if callable(self._roi_spec):
            return np.asarray(self._roi_spec(reduc, self.spacerdi), dtype=bool)
        return make_region_of_interest_mask(reduc, self.spacerdi, **self._roi_spec)

    def _resolve_sigma(self, reduc):
        weighting = self.err_weighting
        if callable(weighting):
            sigma = weighting(reduc)
        elif isinstance(weighting, (bool, np.bool_)):
            if not weighting:
                return 1
            if reduc.err_rolls is None:
                raise ValueError("err_weighting=True requires the data reduction to include uncertainties (prop_err=True for the SpaceRDI object); otherwise set err_weighting=False.")
            sigma = reduc.err_rolls
        else:
            sigma = np.asarray(weighting)
        if (np.ndim(sigma) != 0) and (np.shape(sigma) != reduc.rolls.shape):
            raise ValueError(f"The err_weighting array has shape {np.shape(sigma)}, which doesn't match the shape of the data reduction ({reduc.rolls.shape}). "
                             "If the shape of the reduction changes (e.g., between prepare_fit and finalize), provide a function of the reduction instead.")
        return sigma

    def _state_snapshot(self):
        sp = self.spacerdi
        return dict(concat=sp.concat,
                    crop=tuple(np.asarray(sp._crop_indices).tolist()),
                    data=(sp._imcube_sci, sp._imcube_ref),
                    zones=(sp._optzones, sp._subzones),
                    rdi_settings=dict(sp.rdi_settings))

    def _state_changes(self):
        """Lists what has changed about the SpaceRDI object since the data for the fit were prepared."""
        old, new = self._state, self._state_snapshot()
        changes = []
        if new['concat'] != old['concat']:
            changes.append('the loaded concatenation')
        if new['crop'] != old['crop']:
            changes.append('the crop')
        if any(a is not b for a, b in zip(new['data'], old['data'])):
            changes.append('the data arrays')
        if any(a is not b for a, b in zip(new['zones'], old['zones'])):
            changes.append('the optimization/subtraction zones')
        s_old, s_new = old['rdi_settings'], new['rdi_settings']
        if (s_old.keys() != s_new.keys()) or any(s_old[key] is not s_new[key] for key in s_old):
            changes.append('the RDI settings (e.g., after mcrdi_presets or hpfrdi_presets)')
        return changes

    def _set_data(self, rdi_reduc):
        self.rdi_reduc = rdi_reduc
        self.roi = self._make_roi(rdi_reduc)
        self._sigma = self._resolve_sigma(rdi_reduc)
        self._state = self._state_snapshot()
        self.clear_cache()

    def _prepare_components(self):
        ctx = self._make_context(self.params.valuesdict())
        for component in self.components:
            component.prepare(ctx, self.params)

    def _check_ready(self):
        if self.rdi_reduc is None:
            raise RuntimeError("The data have not been prepared; run prepare_fit before evaluating or fitting the model.")
        changes = self._state_changes()
        if len(changes) != 0:
            raise RuntimeError(f"Since prepare_fit (or prepare_final) was run, there have been changes to {' and '.join(changes)} of the SpaceRDI object. "
                               "The prepared data no longer match its state, so forward models would not be comparable to them; please run prepare_fit again.")

    ###########################################################################
    # Forward modeling
    ###########################################################################

    def _make_context(self, pdict):
        sp = self.spacerdi
        dist_key = tuple(pdict[key] for key in self._dist_keys)
        if len(dist_key) == 0:
            siaf_aper, ptsrc_ext, npad = sp.siaf_aper, 2, 0
        else:
            if self._siaf_memo[0] == (sp.concat, dist_key):
                siaf_aper = self._siaf_memo[1]
            else:
                siaf_aper = get_siaf_aper(sp._aperturename, sp._instrument, Sci2Idl_coeffs=dict(zip(self._dist_keys, dist_key)))
                self._siaf_memo = ((sp.concat, dist_key), siaf_aper)
            ptsrc_ext, npad = 0, 3
        return ModelContext(spacerdi=sp, distance=self.distance, siaf_aper=siaf_aper, distorted_coords=self.distorted_coords,
                            rmax_accuracy=self.rmax_accuracy, ptsrc_ext=ptsrc_ext, npad=npad, dist_key=dist_key, verbose=self.verbose)

    def _realize(self, component, pdict, ctx, eps):
        """Generates the model of a component and forward models it (unless it's unchanged since the last evaluation)."""
        key = (component.cache_key(pdict), (ctx.dist_key if component.uses_distortion else None))
        cached = self._cache.get(component.label)
        if (cached is not None) and (cached['key'] == key):
            return cached

        sp = self.spacerdi
        output = component.generate(ctx, pdict)
        if isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError(f"The generate method of '{component.label}' should return either the model cube or a tuple (modelcube, raw_model); got a tuple of length {len(output)}.")
            modelcube, raw_model = output
        else:
            modelcube, raw_model = output, None
        modelcube = np.array(modelcube, copy=True)
        if modelcube.shape != sp.imcube_sci.shape:
            raise ValueError(f"The model cube of component '{component.label}' has shape {modelcube.shape}, but should match the data: {sp.imcube_sci.shape}.")
        modelcube[np.isnan(sp.imcube_sci)] = np.nan

        sp.set_circumstellar_model(modelcube+eps)
        fmreduc = sp.run_rdi(forward_model=True, save_products=False, collapse_rolls=True,
                             derotate=self.rdi_reduc.derotated, correct_distortion=self.rdi_reduc.distortion_corrected)

        entry = dict(key=key, modelcube=modelcube, raw_model=raw_model, fmreduc=fmreduc)
        self._cache[component.label] = entry
        return entry

    def _forward_model(self, params, add_fm_eps=None):
        """
        Generates and forward models the components, solves for the coefficients of the autoscaled
        components, and returns all of the (unscaled) pieces needed to build a solution or evaluate the residuals.
        """
        self._check_ready()
        rdi_reduc = self.rdi_reduc
        sp = self.spacerdi
        pdict = params.valuesdict()
        ctx = self._make_context(pdict)

        if add_fm_eps is None:
            add_fm_eps = self.add_fm_eps
        eps = (np.random.normal(scale=1e-32, size=(sp.ny, sp.nx)) if add_fm_eps else 0)

        entries = [self._realize(component, pdict, ctx, eps) for component in self.components]
        autoscale = np.array([component.is_autoscaled(params) for component in self.components], dtype=bool)
        coeffs = np.array([(1. if auto else pdict[component.pname('coeff')]) for component, auto in zip(self.components, autoscale)], dtype=float)

        fmimages = np.array([entry['fmreduc'].rolls for entry in entries])
        sig = self._sigma

        # Fixed model components
        if np.any(~autoscale):
            fmrdi_fixed = np.tensordot(coeffs[~autoscale], fmimages[~autoscale], axes=(0, 0))
        else:
            fmrdi_fixed = 0

        # Autoscaled model components
        if np.any(autoscale):
            scale_optzone = (np.all(self.roi, axis=0) if self.roi.ndim == 3 else self.roi)
            autoscale_coeffs = compute_rdi_coefficients((rdi_reduc.rolls - fmrdi_fixed)[np.newaxis, :]/sig,
                                                        fmimages[autoscale]/sig,
                                                        [scale_optzone])
            coeffs[autoscale] = np.median(autoscale_coeffs, axis=(0, 1, 3))

        fmrdi_rolls = np.tensordot(coeffs, fmimages, axes=(0, 0))
        return dict(entries=entries, autoscale=autoscale, coeffs=coeffs, fmrdi_rolls=fmrdi_rolls)

    def _as_params(self, params):
        if params is None:
            return self.params
        return getattr(params, 'params', params) # accepts an lmfit MinimizerResult as well

    def residuals(self, params):
        """
        The objective function: returns the (weighted) absolute residuals between the
        data and the forward modeled composite model for the pixels in the
        region of interest. This is the function that fit minimizes, and can be
        used with lmfit directly. Requires prepare_fit to have been run.

        Parameters
        ----------
        params : lmfit.Parameters
            Parameters of the model (as in self.params) at which to evaluate the model.
        """
        fm = self._forward_model(params)
        res = ((self.rdi_reduc.rolls - fm['fmrdi_rolls'])/self._sigma)[..., self.roi]

        if self.q_clip is None:
            return np.abs(res)

        low, upp = np.nanpercentile(res, self.q_clip)
        return np.abs(res[(res >= low) & (res <= upp)])

    def evaluate(self, params=None, add_fm_eps=None, set_model=True):
        """
        Evaluates the model for a set of parameters and returns all of the resulting
        products as a ModelSolution. Requires prepare_fit to have been run.

        Parameters
        ----------
        params : lmfit.Parameters, optional
            Parameters of the model (as in self.params) at which to evaluate the model. 
            Defaults to self.params. An lmfit MinimizerResult (as returned by fit) is also accepted.

        add_fm_eps : bool, optional
            Overrides the add_fm_eps setting of the model if provided.

        set_model : bool
            If True, sets the composite model as the circumstellar model of the SpaceRDI object,
            so that it can be used directly for MCRDI, saving the model, etc.
        """
        params = self._as_params(params)
        self._check_params(params)
        fm = self._forward_model(params, add_fm_eps=add_fm_eps)

        rdi_reduc = self.rdi_reduc
        derotated = rdi_reduc.derotated
        labels = self.labels
        coeffs = {}
        autoscale = {}
        modelcubes = {}
        fmreducs = {}
        raw_models = {}

        fmrdi_reduc = deepcopy(fm['entries'][0]['fmreduc'])
        if derotated:
            fmrdi_reduc.im = fmrdi_reduc.im*0

        for i, label in enumerate(labels):
            entry = fm['entries'][i]
            coeff = float(fm['coeffs'][i])
            coeffs[label] = coeff
            autoscale[label] = bool(fm['autoscale'][i])
            modelcubes[label] = entry['modelcube']*coeff
            if entry['raw_model'] is not None:
                raw_models[label] = entry['raw_model']*coeff

            fmreduc = deepcopy(entry['fmreduc'])
            if fmreduc.rolls is not None:
                fmreduc.rolls = fmreduc.rolls*coeff
            if derotated:
                fmreduc.im = fmreduc.im*coeff
                fmrdi_reduc.im = fmrdi_reduc.im + fmreduc.im
            fmreduc.update_reduc_label(f'{fmreduc.reduc_label}, {label}')
            fmreduc.update_output_ext(f'{rdi_reduc.output_ext}_{label}_fwdmod')
            fmreducs[label] = fmreduc

        fmrdi_reduc.rolls = fm['fmrdi_rolls'].astype(rdi_reduc.rolls.dtype)
        modelcube_composite = np.sum(list(modelcubes.values()), axis=0)

        if set_model:
            self.spacerdi.set_circumstellar_model(modelcube_composite)

        if isinstance(self.err_weighting, (bool, np.bool_)):
            err_weighting = bool(self.err_weighting)
        else:
            err_weighting = ('callable' if callable(self.err_weighting) else 'array')
        settings = dict(q_clip=str(self.q_clip), err_weighting=err_weighting)

        return ModelSolution(params=params.copy(), labels=labels, coeffs=coeffs, autoscale=autoscale, modelcubes=modelcubes,
                             fmreducs=fmreducs, raw_models=raw_models, modelcube_composite=modelcube_composite,
                             fmrdi_reduc=fmrdi_reduc, rdi_reduc=rdi_reduc, roi=self.roi, settings=settings)

    ###########################################################################
    # Optimization and final results
    ###########################################################################

    def fit(self, params=None, method='powell', progress=True, iter_cb=None, **kwargs):
        """
        Optimizes the model parameters with lmfit.minimize, minimizing the
        residuals returned by the residuals method. Requires prepare_fit to
        have been run.

        Parameters
        ----------
        params : lmfit.Parameters, optional
            Initial parameters (and which of them vary). Defaults to self.params.

        method : str
            The fitting method (see lmfit.minimize).

        progress : bool
            If True, displays the number of models that have been evaluated.

        iter_cb : callable, optional
            Callback function passed to lmfit.minimize (called after every evaluation of the model).

        kwargs : 
            Additional keyword arguments for lmfit.minimize. By default, nan_policy='omit'.

        Returns
        -------
        result : lmfit.minimizer.MinimizerResult
            The result of the optimization; result.params contains the best-fit parameters.
        """
        params = self._as_params(params)
        self._check_params(params)
        self._check_ready()
        kwargs.setdefault('nan_policy', 'omit')
        n_evaluated = [0]

        def callback(par, it, resid, *args, **kws):
            n_evaluated[0] = it
            if progress:
                print('Models evaluated: {0: <16}'.format(it), end='\r')
            if iter_cb is not None:
                return iter_cb(par, it, resid, *args, **kws)

        result = lmfit.minimize(self.residuals, params, method=method, iter_cb=callback, **kwargs)
        if progress:
            print('Models evaluated: {0: <16}'.format(n_evaluated[0]))
        return result

    def finalize(self, params, save=True, roi=None, rmax_accuracy=None, rdi_presets=True):
        """
        Produces the final results for a set of parameters (generally the
        best-fit result of the optimization): undoes any cropping and carries out
        the RDI reduction of the full frame with derotation and distortion correction
        (see prepare_final), evaluates the model for it, and creates reductions
        of each component and of the composite model without forward modeling
        (as for SpaceRDI.make_model_reduc). 
        
        If save is True, the following are saved to the output directory of the
        SpaceRDI object (the name of each file includes the output_ext given in
        parentheses, assuming the standard RDI presets; 'x' is the label of a component):

          - the RDI reduction of the data ('rdi_psfsub'),
          - for each component and for the composite model: the forward modeled
            reduction ('rdi_psfsub_x_fwdmod'; 'rdi_psfsub_fwdmod' for the
            composite), the reduction of the model without forward modeling
            ('cssmodel_x'; 'cssmodel'), and the model cube for each exposure
            (also 'cssmodel_x'; 'cssmodel', in files named like the data
            files),
          - an unconvolved model that sums any components that have one ('rawmodel').

        The header of each of the model products includes the model_dict of the
        solution (see ModelSolution.model_dict). 

        The circumstellar model of the SpaceRDI object is left set to the
        composite model, so e.g. MCRDI can follow directly.

        Parameters
        ----------
        params : lmfit.Parameters
            The parameters for the model. An lmfit MinimizerResult (as returned by fit) is also accepted.

        save : bool
            If True, saves the products to disk.

        roi : dict or callable, optional
            The region of interest definition (see prepare_fit). Defaults to the
            one that was used by prepare_fit.

        rmax_accuracy : float, optional
            The largest separation (in au) at which to calculate disk models.
            Defaults to None (the edge of the field of view).

        rdi_presets : bool
            See prepare_fit.

        Returns
        -------
        solution : ModelSolution
            With the additional modelreducs, model_reduc, and raw_reduc attributes.
        """
        params = self._as_params(params)
        self._check_params(params)
        sp = self.spacerdi

        self.prepare_final(roi=roi, rmax_accuracy=rmax_accuracy, rdi_presets=rdi_presets, save_rdi=save)
        sol = self.evaluate(params, add_fm_eps=False, set_model=True)
        model_dict = sol.model_dict()

        modelreducs = {}
        for label, modelcube in sol.modelcubes.items():
            output_ext = f'cssmodel_{label}'
            sp.set_circumstellar_model(modelcube)
            if save:
                sp.save_circumstellar_model(output_ext=output_ext, model_dict=model_dict)
            modelreducs[label] = sp.make_model_reduc(output_ext=output_ext, reduc_label=f'CSS Model, {label}', match_reduc=sol.rdi_reduc, model_dict=model_dict)

        sp.set_circumstellar_model(sol.modelcube_composite)
        if save:
            sp.save_circumstellar_model(model_dict=model_dict)
        sol.model_reduc = sp.make_model_reduc(match_reduc=sol.rdi_reduc, model_dict=model_dict)
        sol.modelreducs = modelreducs

        if sol.raw_model is not None:
            sol.raw_reduc = SpaceReduction(spacerdi=sp, im=sol.raw_model, c_star_out=sp.c_star, c_coron_out=sol.rdi_reduc.c_coron,
                                           output_ext='rawmodel', reduc_label='Raw Model')
            sol.raw_reduc.primary_header.update(model_dict)

        sol.fmrdi_reduc.primary_header.update(model_dict)
        for fmreduc in sol.fmreducs.values():
            fmreduc.primary_header.update(model_dict)

        if save:
            products = [sol.fmrdi_reduc, sol.model_reduc, *sol.fmreducs.values(), *modelreducs.values()]
            if sol.raw_reduc is not None:
                products.append(sol.raw_reduc)
            for reduc in products:
                reduc.save(overwrite=sp.overwrite)

        return sol
